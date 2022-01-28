"""This module is responsible for fetching the webpages"""
# pylint: disable=too-few-public-methods
import json
import time
import base64
import logging
import asyncio
import itertools
import abc
import re
import urllib.parse
from abc import abstractmethod
from typing import (
    Any, Dict, Generator, Optional, TypeVar, Iterable, List, AsyncIterable,
    Tuple, Sequence, Union
)
from collections import Counter
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
    if protocol.startswith('h3-'):
        return protocol
    match = re.match(r'Q(\d{3})', protocol)
    if match:
        return 'QUIC_VERSION_{}'.format(int(match[1]))
    raise ValueError(f"Invalid protocol string: {protocol}")


def options_for_quic(
    url: str, protocol: str, force_quic_on_all: bool = False,
) -> List[str]:
    """Return the options necessary for Chromium to request a particular
    domain with or without QUIC.

    The flag force_quic_on_all will force QUIC on dependent resources
    when the protocol is a QUIC protocol.
    """
    if protocol.lower() == "tcp":
        return ["--disable-quic"]

    parsed = urllib.parse.urlparse(url, scheme="https")
    assert parsed.port is None
    assert parsed.hostname is not None

    options = ["--enable-quic"]
    if force_quic_on_all:
        options.append("--origin-to-force-quic-on=*")
    else:
        options.append(f"--origin-to-force-quic-on={parsed.hostname}:443")

    if protocol.lower() != "quic":
        version_string = _chromium_quic_version_string(protocol)
        options += [f"--quic-version={version_string}"]
    return options


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

    force_quic_on_all :
        Force QUIC on dependent resources when creating a Chromium
        instance for QUIC.
    """
    def __init__(
        self, driver_path: str = './chromedriver',
        max_attempts: int = 3, retry_delay: int = 2,
        force_quic_on_all: bool = False,
    ):
        assert driver_path
        assert max_attempts > 0
        self.driver_path = driver_path
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay
        self.force_quic_on_all = force_quic_on_all
        self._logger = logging.getLogger(__name__)

    def create(self, url: str, protocol: str) -> WebDriver:
        options = self.chrome_options(url, protocol)
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
        assert False, "Should never reach here"

    def chrome_options(self, url: str, protocol: str) \
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

        for argument in options_for_quic(
            url, protocol, force_quic_on_all=self.force_quic_on_all
        ):
            options.add_argument(argument)

        # Add tracing of network requests
        options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
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
        self._performance_log = None

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
        if self._performance_log is None:
            log = self.driver.get_log('performance')
            assert log is not None

            for entry in log:
                entry['message'] = json.loads(entry['message'])

            # We filter Page events as Chrome doesnt properly
            log = [entry for entry in log if not
                   entry['message']['message']['method'].startswith('Page.')]
            self._performance_log = log

        return self._performance_log

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


def encode_result(result: Result) -> str:
    """Encode the result in json format."""
    result_copy = result
    if result_copy['packets'] is not None:
        result_copy = result_copy.copy()
        result_copy['packets'] = base64.b64encode(  # type: ignore
            result_copy['packets']).decode('utf8')
    return json.dumps(result_copy)


def decode_result(json_str: str, full_decode: bool = True) -> Result:
    """Decode the json string to a result object.

    If full_decode is false, 'packets' and 'http_trace' are set to their
    empty values.
    """
    result = json.loads(json_str)

    if not full_decode:
        result['http_trace'] = []
        result['packets'] = b''
    else:
        result['packets'] = base64.b64decode(result["packets"])
    return result


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
            logger.info('Successfully fetched %s [%s].', url, protocol)
        except FetchTimeout as error:
            result['status'] = 'timeout'
            logger.info('%s [%s]: %s', url, protocol, error)
        except (
            FetchFailed, UnexpectedAlertPresentException, WebDriverException
        ) as error:
            result['status'] = 'failure'
            logger.info('Failed to fetch %s [%s]: %s', url, protocol, error)
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
    attempts_per_protocol :
        After max_attempts failures on a protocol, drop that protocol
        but continue to other protocols.
    delay :
        Wait delay seconds between each successive attempt for a given
        URL.  A delay of 0 means do not wait.
    """
    def __init__(
        self, sniffer: PacketSniffer, session_factory: SessionFactory,
        max_attempts: int = 3, attempts_per_protocol: bool = False,
        delay: float = 0
    ):
        assert delay >= 0
        assert max_attempts > 0
        self.max_attempts: Final = max_attempts
        self.attempts_per_protocol = attempts_per_protocol
        self.delay: Final = delay
        self._sniffer = sniffer
        self._session_factory = session_factory
        self._lock: Optional[asyncio.Lock] = None
        self._logger = logging.getLogger(__name__)

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
            status = result["status"]
            yield result

            if status == 'success':
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
                if self.attempts_per_protocol:
                    self._logger.info(
                        "Stopping collection of %s (%s) as it has "
                        "failed %d times sequentially.", url, protocol,
                        self.max_attempts)
                    remaining[protocol] = 0
                else:
                    self._logger.info(
                        "Stopping collection of %s (all protocols) as it has "
                        "failed %d times sequentially.", url, self.max_attempts)
                    return
            else:
                remaining[protocol] -= 1

            if sum(remaining.values()) == 0:
                return
            immediate = False

    async def sample_multiple(
        self,
        urls: Union[Dict[str, Dict[str, int]],
                    Tuple[Sequence[str], Dict[str, int]]]
    ) -> AsyncIterable[Result]:
        """Samples multiple URLs and yields results as soon as they are
        available.

        Parameters
        ----------
        urls:
            Either a dictionary mapping each URL to a Counter
            identifying how many times each protocol version should be
            collected for that URL, or a tuple of the URLs and a single
            counter mapping protocol versions to repetitions.
        """
        from itertools import repeat
        result_stream = aiostream.stream.merge(
            *[self.sample_url(url, protocols)
              for url, protocols in
              (zip(urls[0], repeat(urls[1])) if isinstance(urls, tuple)
               else urls.items())])

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


def filter_by_checkpoint(
    urls: List[str], checkpoint: Iterable[Result], counter: Dict[str, int],
    max_attempts: int = 3
) -> Dict[str, Counter]:
    """Filter the URLs by the checkpoint.

    Return a dictionary mapping the URLs still to be collected to the number of
    each protocol to be collected.

    The returned counter is guaranteed to not have any negative elements.
    The returned dict will not have any URLs with empty counters.
    URLs which failed at least max_attempts times in sequence will also
    be removed.
    """
    failures: Counter = Counter()
    base = {url: Counter(counter) for url in urls}
    checkpoint_results: Dict[str, Counter] = {url: Counter() for url in urls}

    for result in checkpoint:
        url = result['url']

        if failures[url] >= max_attempts:
            continue

        if result['status'] == 'success':
            checkpoint_results[url][result['protocol']] += 1
            failures[url] = 0
        else:
            failures[url] += 1

    for url, completed_counter in checkpoint_results.items():
        # Copy-Subtraction ensures that there are no negative counts
        base[url] = base[url] - completed_counter
        if sum(base[url].values()) == 0 or failures[url] >= max_attempts:
            del base[url]
    return base
