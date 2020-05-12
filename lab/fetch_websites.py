"""This module is responsible for fetching the webpages"""
# pylint: disable=too-few-public-methods
import json
import time
import logging
import asyncio
import itertools
import abc
import re
import urllib.parse
from abc import abstractmethod
from typing import (
    Any, Dict, Generator, Optional, TypeVar, Iterable, List, AsyncIterable,
    Sequence,
)
from urllib3.exceptions import MaxRetryError

from mypy_extensions import TypedDict
from typing_extensions import Literal, Final
import aiostream
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


def _chromium_quic_version_string(protocol: str) -> str:
    if protocol.startswith('h3-Q'):
        return protocol
    match = re.match(r'Q(\d{3})', protocol)
    if match:
        return 'QUIC_VERSION_{}'.format(int(match[1]))
    # Versions h3-25 etc are not currently supported in Chrome 81
    raise ValueError(f"Invalid protocol string: {protocol}")


def options_for_quic(url: str, protocol: str) -> List[str]:
    """Return the options necessary for Chromium to request a particular
    domain with or without QUIC.
    """
    if protocol.lower() == "tcp":
        return ["--disable-quic"]

    version_string = _chromium_quic_version_string(protocol)
    parsed = urllib.parse.urlparse(url, scheme="https")
    assert parsed.port is None
    assert parsed.hostname is not None
    return [f"--origin-to-force-quic-on={parsed.hostname}:443",
            f"--quic-version={version_string}"]


class WebDriverFactory(abc.ABC):
    """Factory for creating WebDrivers."""
    @abstractmethod
    def create(self, url: str, protocol: str) -> WebDriver:
        """Creates a webdriver for requesting the sepecified URL via
        protocol.
        """


class ChromiumFactory(WebDriverFactory):
    """Creates Chromium webdrivers.

    max_attempts, retry_delay :
        Attempt creation of the web-driver at most max_attempts times.
        This helps with spurious failures while running in docker.  Wait
        retry_delay seconds after each attempt.
    """
    def __init__(self, driver_path: str = './chromedriver',
                 max_attempts: int = 3, retry_delay: int = 2):
        assert driver_path
        assert max_attempts > 0
        self.driver_path = driver_path
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self._logger = logging.getLogger(__name__)

    def create(self, url: str, protocol: str) -> WebDriver:
        options = ChromiumFactory.chrome_options(url, protocol)
        for attempt in range(self.max_attempts):
            try:
                driver = webdriver.Chrome(
                    executable_path=self.driver_path, options=options)
                self._logger.info('Chromedriver created with %s and args %s.',
                                  self.driver_path, options.arguments)
                return driver
            except (WebDriverException, MaxRetryError):
                self._logger.critical(
                    "Failed to create a webdriver from %s with args %s. "
                    "Attempt %d of %d.", self.driver_path, options.arguments,
                    attempt, self.max_attempts)
                if attempt == (self.max_attempts - 1):
                    raise
                self._logger.info("Waiting %ss before next attempt",
                                  self.retry_delay)
                time.sleep(self.retry_delay)

    @staticmethod
    def chrome_options(url: str, protocol: str) -> webdriver.ChromeOptions:
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

        for argument in options_for_quic(url, protocol):
            options.add_argument(argument)

        # Add tracing of network requests
        options.set_capability('goog:loggingPrefs', {'performance': 'DEBUG'})
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False,
        })
        return options


class ChromiumSession:
    """A browser session for fetching a webpage."""
    def __init__(
        self, url: str, protocol: str, driver_factory: WebDriverFactory = None
    ):
        self.url = url
        self.protocol = protocol
        self.driver = None
        self._driver_factory = driver_factory or ChromiumFactory()
        self._logger = logging.getLogger(__name__)

    def begin(self) -> None:
        """Starts the session."""
        assert self.driver is None
        self._logger.info('Starting session %s', self)
        self.driver = self._driver_factory.create(self.url, self.protocol)

    def close(self):
        """Ends the session."""
        try:
            if self.driver is not None:
                self.driver.quit()
        except WebDriverException as err:
            self._logger.warning("%s suppressed '%s' on driver quit.",
                                 self, err.msg)
        finally:
            self.driver = None
            self._logger.info('%s closed', self)

    def __str__(self):
        return f'ChromiumSession(url={self.url}, protocol={self.protocol}, ...)'

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def page_source(self) -> str:
        """Returns the source document of the requested page."""
        assert self.driver is not None
        return self.driver.page_source

    @property
    def current_url(self) -> str:
        """Returns the currently loaded url."""
        assert self.driver is not None
        return self.driver.current_url

    def fetch_page(self, timeout: int = 30) -> str:
        """Fetches the webpage and returns the page's source.

        Raises `FetchTimeout` if the page fails to load within `timeout`
        seconds, and `FetchFailed` if the response is empty (due to some error
        such as QUIC not being supported).
        """
        assert self.driver is not None
        self.driver.set_page_load_timeout(timeout)
        self._logger.info('Webpage request timeout set to to %ds.', timeout)

        try:
            self.driver.get(self.url)
            page_source = self.driver.page_source
        except selenium.common.exceptions.TimeoutException as error:
            raise FetchTimeout(self.url, timeout) from error
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
            raise FetchFailed(
                f"Fetch failed for {self.url}. Invalid unicode in the page.")

    def performance_log(self) -> Iterable[dict]:
        """Returns a dictionary of the peformance log messages gathered from the
        browser.
        """
        assert self.driver is not None
        log = self.driver.get_log('performance')
        assert log is not None

        for entry in log:
            entry['message'] = json.loads(entry['message'])
        return log

    def _validate_response(self, page_source: str, min_chars: int = 100):
        """Raises `FetchFailed` error on an empty html document, or
        small documents -- less than min_chars non-whitespace
        characters.
        """
        page_source = page_source.replace(' ', '')

        if page_source == "<html><head></head><body></body></html>":
            raise FetchFailed(
                f"Fetch failed for {self.url}. Server may not support QUIC.")
        if len(page_source) < min_chars:
            raise FetchFailed(
                f"Fetch failed for {self.url}. Document too small "
                f"({len(page_source)} < {min_chars} chars).")

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

        for event in _response_events():
            if event['params']['type'].lower() == 'document':
                response = event['params']['response']
                if (response['status'] >= 400 and self.url in response['url']
                        and len(response['url']) - len(self.url) <= 1):
                    raise FetchFailed(
                        f"Fetch failed for {self.url}. Response code of "
                        f"{response['status']} for document {response['url']}")


class SessionFactory(abc.ABC):
    """Factory class for creating fetch session."""
    @abstractmethod
    def create(self, url: str, protocol: str) -> ChromiumSession:
        """Creates and returns a new ChromiumSession."""


class ChromiumSessionFactory(SessionFactory):
    """Factory class for ChromiumSessions.

    If driver_factory is None, any other kwargs will be passed to a
    new ChromiumFactory.  Otherwise, driver_factory is used and other
    keyword arguments are ignored.
    """
    def __init__(self, driver_factory: ChromiumFactory = None, **factory_kw):
        self._driver_factory = driver_factory or ChromiumFactory(**factory_kw)

    def create(self, url: str, protocol: str) -> ChromiumSession:
        return ChromiumSession(url, protocol, self._driver_factory)


Result = TypedDict('Result', {
    'url': str,
    'protocol': str,
    'page_source': Optional[str],
    'final_url': Optional[str],
    'status': Literal['success', 'timeout', 'failure'],
    'http_trace': Iterable[Dict[str, Any]],
    'packets': bytes,
})

# pylint: disable=line-too-long

# class RepetitionTracker:
#     """Tracks the repetitions and failures in the WebsiteTraceExperiment."""
#     def __init__(self, repetitions: int, max_consecutive_failures: int = 3):
#         assert max_consecutive_failures > 0
#         self.repetitions = repetitions
#         self.max_consecutive_failures = max_consecutive_failures
#         self.counts = {'success': 0, 'total': 0, 'failure-seq': 0}
#
#     def is_first(self) -> bool:
#         """Returns true iff there have been no observed successful results."""
#         return self.counts['success'] == 0
#
#     def only_failures(self) -> bool:
#         """Return true iff there are failures without any successes."""
#         return self.counts['success'] == 0 and self.counts['total'] > 0
#
#     def too_many_failures(self) -> bool:
#         """Returns true iff a sequence of max_consecutive_failures was observed.
#         """
#         return self.counts['failure-seq'] == self.max_consecutive_failures
#
#     def repeat(self) -> bool:
#         """Returns true if the experiment should continue fetching."""
#         assert self.counts['failure-seq'] <= self.max_consecutive_failures
#         if self.only_failures() or self.too_many_failures():
#             return False
#         assert self.counts['success'] <= self.repetitions
#         if self.counts['success'] == self.repetitions:
#             return False
#         return True
#
#     def observe(self, result: Result) -> Result:
#         """Update the repetition progress with the observed result."""
#         if result['status'] == 'success':
#             self.counts['success'] += 1
#             self.counts['failure-seq'] = 0
#         else:
#             self.counts['failure-seq'] += 1
#         self.counts['total'] += 1
#         return result
#
#
# class WebsiteTraceExperiment:
#     """Experiment consisting for fetching and tracing website traffic."""
#     def __init__(self, sniffer: PacketSniffer, session_factory: SessionFactory):
#         self._logger = logging.getLogger(__name__)
#         self._sniffer = sniffer
#         self._session_factory = session_factory
#
#     def _sample_with_repetitions(
#         self,
#         domain: Domain,
#         tracker: RepetitionTracker,
#         use_quic: bool,
#         keep_sources: Literal['all', 'first', 'none'] = 'all'
#     ) -> Iterator[Result]:
#         """Perform the sampling with repetitions using the provided tracker."""
#         protocol = 'QUIC' if use_quic else 'TCP'
#
#         while tracker.repeat():
#             with_source = keep_sources == 'all' or (
#                 keep_sources == 'first' and tracker.is_first())
#             yield tracker.observe(
#                 self.sample(domain, use_quic=use_quic, with_source=with_source))
#
#         if tracker.only_failures() or tracker.too_many_failures():
#             self._logger.warning("Stopped fetching %s sites for %s due to "
#                                  "failures.", protocol, domain)
#         self._logger.info("%s fetch stats - %s", protocol, tracker.counts)
#
#     def sample_domain(
#         self,
#         domain: Domain,
#         repetitions: int = 1,
#         keep_sources: Literal['all', 'first', 'none'] = 'all',
#         fetch_tcp: bool = True
#     ) -> Iterator[Result]:
#         """Yields the results of sampling with and without QUIC."""
#         tracker = RepetitionTracker(repetitions)
#         yield from self._sample_with_repetitions(
#             domain, tracker, use_quic=True, keep_sources=keep_sources)
#
#         if fetch_tcp:
#             tracker = RepetitionTracker(tracker.counts['success'] or 1)
#             yield from self._sample_with_repetitions(
#                 domain, tracker, use_quic=False, keep_sources=keep_sources)
#
#     def sample(
#         self, domain: Domain, use_quic: bool, with_source: bool = True
#     ) -> Result:
#         """Fetch the domain and returns the result."""
#         page_source = None
#         status: Literal['success', 'timeout', 'failure'] = 'success'
#
#         with self._session_factory.create(domain, use_quic) as session:
#             self._sniffer.start()
#             try:
#                 page_source = session.fetch_page()
#                 status = 'success'
#                 self._logger.info('Successfully fetched domain %s (quic: %s).',
#                                   domain, use_quic)
#             except FetchTimeout as error:
#                 self._logger.warning('%s (quic: %s). %s', domain, use_quic,
#                                      error)
#                 status = 'timeout'
#             except (
#                 FetchFailed, UnexpectedAlertPresentException,
#                 WebDriverException,
#             ) as error:
#                 self._logger.warning('Failed to fetch %s (quic: %s). %s',
#                                      domain, use_quic, error)
#                 status = 'failure'
#             finally:
#                 self._sniffer.stop()
#
#             if not with_source:
#                 page_source = None
#
#             return dict(page_source=page_source, status=status,
#                         domain=domain, with_quic=use_quic,
#                         http_trace=session.performance_log(),
#                         packets=self._sniffer.results)
#
#


def collect_trace(url: str, protocol: str, sniffer: PacketSniffer,
                  session_factory: SessionFactory) -> Result:
    """Fetch the URL and return the result of the fetch."""
    logger = logging.getLogger(__name__)
    result: Result = dict(url=url, protocol=protocol, final_url=None,
                          page_source=None, status='success', http_trace=[],
                          packets=b'')

    with session_factory.create(url, protocol) as session:
        sniffer.start()
        try:
            result['page_source'] = session.fetch_page()
            result.update({'final_url': session.current_url,
                           'http_trace': session.performance_log(),
                           'status': 'success'})
            logger.debug('Successfully fetched %s [%s].', url, protocol)
        except FetchTimeout as error:
            result['status'] = 'timeout'
            logger.debug('%s [%s]: %s', url, protocol, error)
        except (
            FetchFailed, UnexpectedAlertPresentException, WebDriverException
        ) as error:
            result['status'] = 'failure'
            logger.debug('Failed to fetch %s [%s]: %s', url, protocol, error)
        finally:
            sniffer.stop()

        if result['status'] == 'success':
            result['packets'] = sniffer.results

        return result


class MaxSamplingAttemptError(RuntimeError):
    """Raised when an upper limit for attempts has been hit."""


class ProtocolSampler:
    """Sample a set of protocols repeatedly.

    Parameters
    ----------
    max_attempts :
        Allow up to max_attempts sequential failures when collecting a
        URL before giving up.
    delay :
        Wait delay seconds between each successive attempt for a given
        URL.  A delay of 0 means do not wait.
    """
    def __init__(
        self, sniffer: PacketSniffer, session_factory: SessionFactory,
        max_attempts: int = 3, delay: float = 0
    ):
        assert delay >= 0
        assert max_attempts > 0
        self.max_attempts: Final = max_attempts
        self.delay: Final = delay
        self._sniffer = sniffer
        self._session_factory = session_factory
        self._lock: Optional[asyncio.Lock] = None

    async def _sample_with_retries(
        self, url: str, protocol: str, immediate: bool = False
    ) -> AsyncIterable[Result]:
        """Attempt to repeatedly sample the protocol. Return False iff
        collection failed due to too many attempts.
        """
        attempts_remaining = self.max_attempts
        while True:
            if not immediate and self.delay > 0:
                await asyncio.sleep(self.delay)

            result = await self.collect_trace(url, protocol)
            yield result

            if result['status'] == 'success':
                return
            attempts_remaining -= 1

            if attempts_remaining == 0:
                raise MaxSamplingAttemptError
            immediate = False

    async def sample_url(self, url: str, protocols: Dict[str, int]) \
            -> AsyncIterable[Result]:
        """Sample a URL repeatedly using the specified protocols."""
        remaining = protocols.copy()
        immediate = True

        for protocol in itertools.cycle(protocols):
            if remaining[protocol] == 0:
                continue

            try:
                coroutine = self._sample_with_retries(url, protocol, immediate)
                async for result in coroutine:
                    yield result
            except MaxSamplingAttemptError:
                return
            else:
                remaining[protocol] -= 1

            if sum(remaining.values()) == 0:
                return
            immediate = False

    async def sample_multiple(
        self, urls: Sequence[str], protocols: Dict[str, int]
    ) -> AsyncIterable[Result]:
        """Samples multiple URLs and yields results as soon as they are
        available.
        """
        result_stream = aiostream.stream.merge(
            *[self.sample_url(url, protocols) for url in urls])

        async with result_stream.stream() as streamer:
            async for result in streamer:
                yield result

    async def collect_trace(self, url: str, protocol: str) -> Result:
        """Run collect_trace asynchrnously. Multiple calls are are
        async-safe, but not thread-safe.
        """
        loop = asyncio.get_running_loop()
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            return await loop.run_in_executor(
                None, collect_trace, url, protocol, self._sniffer,
                self._session_factory)
