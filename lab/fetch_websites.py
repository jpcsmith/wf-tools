"""This module is responsible for fetching the webpages"""
# pylint: disable=too-few-public-methods
import json
import time
import logging
import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    TypeVar,
    Iterable,
    Iterator,
)
from urllib3.exceptions import MaxRetryError

from mypy_extensions import TypedDict
from typing_extensions import Literal
import selenium
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    WebDriverException,
)

from lab.sniffer import PacketSniffer

# Meta type variable for generic types
T = TypeVar('T')  # pylint: disable=invalid-name

# A generator without any input or return values
SimpleGenerator = Generator[T, None, None]


# --------------------------------------------------
# Custom Exceptions
# --------------------------------------------------
class FetchTimeout(Exception):
    """Raised when requesting a website times out."""
    def __init__(self, url: str, timeout: int):
        super().__init__(f"Failed to fetch {url} within {timeout}s")


class FetchFailed(Exception):
    """Raised when requesting a website fails due to an empty result."""


@dataclass(frozen=True)
class Domain:
    """An internet domain name, such as 'example.com' or 'mail.google.com'.
    """
    name: str

    def __post_init__(self):
        assert not self.name.startswith(('http://', 'https://'))

    def as_https_url(self) -> str:
        """Returns the https url for the domain, e.g. https://example.com"""
        return f'https://{self.name}'

    def with_port(self, port: int) -> str:
        """Returns the domain:port representation, e.g. example.com:443"""
        assert 0 <= port < 2**16
        return f'{self.name}:{port}'


class WebDriverFactory(abc.ABC):
    """Factory for creating WebDrivers."""
    @abstractmethod
    def create(self, quic_domain: Optional[Domain]) -> WebDriver:
        """Creates a webdriver for requesting the page associated with the
        optional QUIC domain. If no domain is present, then TCP is assumed.
        """


class ChromiumFactory(WebDriverFactory):
    """Creates Chromium webdrivers."""
    def __init__(self, driver_path: str = './chromedriver',
                 max_attempts: int = 3, retry_delay: int = 3):
        assert driver_path
        assert max_attempts > 0
        self._logger = logging.getLogger(__name__)
        self.driver_path = driver_path
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

    def create(self, quic_domain: Optional[Domain]) -> WebDriver:
        options = self.chrome_options(quic_domain)
        for attempt in range(0, self.max_attempts):
            try:
                driver = webdriver.Chrome(executable_path=self.driver_path,
                                          options=options)
                self._logger.info('Chromedriver created with %s and args %s.',
                                  self.driver_path, options.arguments)
                return driver
            except (WebDriverException, MaxRetryError):
                self._logger.critical(
                    "Failed to create a webdriver from %s with args %s. "
                    "Attempt %d of %d.", self.driver_path, options.arguments,
                    attempt, self.max_attempts)
                if attempt == self.max_attempts:
                    raise
                self._logger.info("Waiting %ss before next attempt",
                                  self.retry_delay)
                time.sleep(self.retry_delay)

    @staticmethod
    def chrome_options(quic_domain: Optional[Domain])\
            -> webdriver.ChromeOptions:
        """Provide a set of options for the chrome webdriver."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')

        # Necessary when running as root
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-setuid-sandbox')

        options.add_argument('--log-level=0')
        options.add_argument('--enable-logging')
        options.add_argument('--v=1')

        # Force or disable QUIC
        if quic_domain is not None:
            options.add_argument('--enable-quic')
            options.add_argument('--quic-version=QUIC_VERSION_43')
            options.add_argument('--origin-to-force-quic-on={}'.format(
                quic_domain.with_port(443)))
        else:
            options.add_argument('--disable-quic')

        # Add tracing of network requests
        options.set_capability('goog:loggingPrefs', {'performance': 'DEBUG'})
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False,
        })
        return options


class ChromiumSession:
    """A browser session for fetching a webpage.

    A separate session is used for each domain to prevent caching and to force
    QUIC on the provided domain.
    """
    def __init__(self, domain: Domain, use_quic: bool,
                 driver_factory: WebDriverFactory):
        self._logger = logging.getLogger(__name__)
        self._domain = domain
        self._driver: WebDriver = None
        self._driver_factory = driver_factory
        self.use_quic = use_quic

    @property
    def page_source(self) -> str:
        """Returns the source document of the requested page."""
        assert self._driver is not None
        return self._driver.page_source

    def begin(self) -> None:
        """Starts the session."""
        assert self._driver is None
        self._logger.info('Starting webpage session for %s', self._domain)
        self._driver = self._driver_factory.create(
            self._domain if self.use_quic else None)

    def close(self) -> None:
        """Ends the session."""
        try:
            self._driver.quit()
        except WebDriverException as err:
            if 'failed to close window in' not in err.msg:
                raise
            self._logger.warning(
                "Error '%s' suppressed on driver quit.", err.msg)

        self._driver = None
        self._logger.info('Session for %s closed', self._domain)

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def fetch_page(self, timeout: int = 30) -> str:
        """Fetches the webpage and returns the page's source.

        Raises `FetchTimeout` if the page fails to load within `timeout`
        seconds, and `FetchFailed` if the response is empty (due to some error
        such as QUIC not being supported).
        """
        self._driver.set_page_load_timeout(timeout)
        self._logger.info('Webpage request timeout set to to %ds.', timeout)

        page_url = self._domain.as_https_url()
        try:
            self._driver.get(page_url)
            page_source = self._driver.page_source
        except selenium.common.exceptions.TimeoutException as error:
            raise FetchTimeout(page_url, timeout) from error
        except WebDriverException as error:
            self._maybe_wrap_error(error)
            raise

        self._validate_response(page_source)
        self._validate_http_status(self.performance_log())
        return page_source

    def _maybe_wrap_error(self, error: WebDriverException):
        """Wraps specific errors in FetchFailed and raises it or does nothing
        otherwise.
        """
        if "bad inspector message" in error.msg[:50]:
            page_url = self._domain.as_https_url()
            raise FetchFailed(f"Fetch failed for {page_url}. Invalid unicode "
                              f"in the page.")

    def performance_log(self) -> Iterable[dict]:
        """Returns a dictionary of the peformance log messages gathered from the
        browser.
        """
        assert self._driver is not None
        log = self._driver.get_log('performance')
        assert log is not None

        for entry in log:
            entry['message'] = json.loads(entry['message'])
        return log

    def _validate_response(self, page_source: str) -> None:
        """Raises `FetchFailed` error on an empty html document."""
        if page_source == "<html><head></head><body></body></html>":
            page_url = self._domain.as_https_url()
            raise FetchFailed(
                f"Fetch failed for {page_url}. Server may not support QUIC.")

    def _validate_http_status(self, log: Iterable[dict]) -> None:
        """Checks the HTTP status in the logs for failure.

        Raises `FetchFailed` iff the browser log contain a request for the
        domain with an HTTP error status.
        """
        def _response_events():
            for entry in log:
                event = entry['message']['message']
                if event['method'] == 'Network.responseReceived':
                    yield event

        url = self._domain.as_https_url()
        for event in _response_events():
            if event['params']['type'].lower() == 'document':
                response = event['params']['response']
                if response['status'] >= 400 and url in response['url'] and (
                        len(response['url']) - len(url) <= 1):
                    raise FetchFailed(
                        f"Fetch failed for {url}. Response code of "
                        f"{response['status']} for document {response['url']}")


class SessionFactory(abc.ABC):
    """Factory class for creating fetch session."""
    @abstractmethod
    def create(self, domain: Domain, with_quic: bool) -> ChromiumSession:
        """Creates and returns a new BrowserSession."""


class ChromiumSessionFactory(SessionFactory):
    """Factory class for ChromiumSessions"""
    def __init__(self, driver_factory: ChromiumFactory):
        self._driver_factory = driver_factory

    def create(self, domain: Domain, with_quic: bool) -> ChromiumSession:
        """Creates and returns a new BrowserSession."""
        return ChromiumSession(domain, with_quic, self._driver_factory)


Result = TypedDict('Result', {
    'domain': Domain,
    'with_quic': bool,
    'page_source': Optional[str],
    'status': Literal['success', 'timeout', 'failure'],
    'http_trace': Iterable[Dict[str, Any]],
    'packets': bytes,
})


class RepetitionTracker:
    """Tracks the repetitions and failures in the WebsiteTraceExperiment."""
    def __init__(self, repetitions: int, max_consecutive_failures: int = 3):
        assert max_consecutive_failures > 0
        self.repetitions = repetitions
        self.max_consecutive_failures = max_consecutive_failures
        self.counts = {'success': 0, 'total': 0, 'failure-seq': 0}

    def is_first(self) -> bool:
        """Returns true iff there have been no observed successful results."""
        return self.counts['success'] == 0

    def only_failures(self) -> bool:
        """Return true iff there are failures without any successes."""
        return self.counts['success'] == 0 and self.counts['total'] > 0

    def too_many_failures(self) -> bool:
        """Returns true iff a sequence of max_consecutive_failures was observed.
        """
        return self.counts['failure-seq'] == self.max_consecutive_failures

    def repeat(self) -> bool:
        """Returns true if the experiment should continue fetching."""
        assert self.counts['failure-seq'] <= self.max_consecutive_failures
        if self.only_failures() or self.too_many_failures():
            return False
        assert self.counts['success'] <= self.repetitions
        if self.counts['success'] == self.repetitions:
            return False
        return True

    def observe(self, result: Result) -> Result:
        """Update the repetition progress with the observed result."""
        if result['status'] == 'success':
            self.counts['success'] += 1
            self.counts['failure-seq'] = 0
        else:
            self.counts['failure-seq'] += 1
        self.counts['total'] += 1
        return result


class WebsiteTraceExperiment:
    """Experiment consisting for fetching and tracing website traffic."""
    def __init__(self, sniffer: PacketSniffer, session_factory: SessionFactory):
        self._logger = logging.getLogger(__name__)
        self._sniffer = sniffer
        self._session_factory = session_factory

    def _sample_with_repetitions(
        self,
        domain: Domain,
        tracker: RepetitionTracker,
        use_quic: bool,
        keep_sources: Literal['all', 'first', 'none'] = 'all'
    ) -> Iterator[Result]:
        """Perform the sampling with repetitions using the provided tracker."""
        protocol = 'QUIC' if use_quic else 'TCP'

        while tracker.repeat():
            with_source = keep_sources == 'all' or (
                keep_sources == 'first' and tracker.is_first())
            yield tracker.observe(
                self.sample(domain, use_quic=use_quic, with_source=with_source))

        if tracker.only_failures() or tracker.too_many_failures():
            self._logger.warning("Stopped fetching %s sites for %s due to "
                                 "failures.", protocol, domain)
        self._logger.info("%s fetch stats - %s", protocol, tracker.counts)

    def sample_domain(
        self,
        domain: Domain,
        repetitions: int = 1,
        keep_sources: Literal['all', 'first', 'none'] = 'all'
    ) -> Iterator[Result]:
        """Yields the results of sampling with and without QUIC."""
        tracker = RepetitionTracker(repetitions)
        yield from self._sample_with_repetitions(
            domain, tracker, use_quic=True, keep_sources=keep_sources)

        tracker = RepetitionTracker(tracker.counts['success'] or 1)
        yield from self._sample_with_repetitions(
            domain, tracker, use_quic=False, keep_sources=keep_sources)

    def sample(
        self, domain: Domain, use_quic: bool, with_source: bool = True
    ) -> Result:
        """Fetch the domain and returns the result."""
        page_source = None
        status: Literal['success', 'timeout', 'failure'] = 'success'

        with self._session_factory.create(domain, use_quic) as session:
            self._sniffer.start()
            try:
                page_source = session.fetch_page()
                status = 'success'
                self._logger.info('Successfully fetched domain %s (quic: %s).',
                                  domain, use_quic)
            except FetchTimeout as error:
                self._logger.warning('%s (quic: %s). %s', domain, use_quic,
                                     error)
                status = 'timeout'
            except (
                FetchFailed, UnexpectedAlertPresentException,
                WebDriverException,
            ) as error:
                self._logger.warning('Failed to fetch %s (quic: %s). %s',
                                     domain, use_quic, error)
                status = 'failure'
            finally:
                self._sniffer.stop()

            if not with_source:
                page_source = None

            return dict(page_source=page_source, status=status,
                        domain=domain, with_quic=use_quic,
                        http_trace=session.performance_log(),
                        packets=self._sniffer.results)
