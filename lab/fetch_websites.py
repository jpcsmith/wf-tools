"""This module is responsible for fetching the webpages"""
# pylint: disable=too-few-public-methods
import io
import abc
from abc import abstractmethod
import time
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    TypeVar,
)

from mypy_extensions import TypedDict
from typing_extensions import Literal
import scapy.sendrecv
import selenium
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    WebDriverException,
)

# Meta type variable for generic types
T = TypeVar('T')  # pylint: disable=invalid-name

# A generator without any input or return values
SimpleGenerator = Generator[T, None, None]


# --------------------------------------------------
# Custom Exceptions
# --------------------------------------------------
class FetchTimeout(Exception):
    """Raised when requesting a website times out."""


class FetchFailed(Exception):
    """Raised when requesting a website fails due to an empty result."""


@dataclass(frozen=True)
class Domain:
    """An internet domain name, such as 'example.com' or 'mail.google.com'.
    """
    name: str

    def __post_init__(self):
        assert 'www' not in self.name and 'http' not in self.name

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
    def __init__(self, driver_path: str = './chromedriver'):
        assert driver_path
        self._logger = logging.getLogger(__name__)
        self.driver_path = driver_path

    def create(self, quic_domain: Optional[Domain]) -> WebDriver:
        options = self.chrome_options(quic_domain)
        try:
            driver = webdriver.Chrome(
                executable_path=self.driver_path, options=options)
        except WebDriverException:
            self._logger.critical(
                "Failed to create a webdriver from %s with arguments %s",
                self.driver_path, options.arguments)
            raise

        self._logger.info('Chromedriver created from %s with arguments %s.',
                          self.driver_path, options.arguments)
        return driver

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
        self._driver.close()
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
        except selenium.common.exceptions.TimeoutException as error:
            raise FetchTimeout(
                f'Failed to fetch {page_url} within {timeout}s.') from error

        self._validate_response(self._driver.page_source)
        return self._driver.page_source

    def performance_log(self) -> dict:
        """Returns a dictionary of the peformance log messages gathered from the
        browser.
        """
        assert self._driver is not None
        return self._driver.get_log('performance')

    def _validate_response(self, page_source: str) -> None:
        """Raises `FetchFailed` error on an empty html document."""
        if page_source == "<html><head></head><body></body></html>":
            page_url = self._domain.as_https_url()
            raise FetchFailed(
                f"Fetch failed for {page_url}. Server may not support QUIC.")


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


class PacketSniffer:
    """Class for capturing network traffic."""
    start_delay = 0.01

    def __init__(self, capture_filter: str = 'tcp or udp port 443'):
        self._logger = logging.getLogger(__name__)
        self._filter = capture_filter
        self._sniffer = scapy.sendrecv.AsyncSniffer(filter=capture_filter)

    @property
    def results(self) -> scapy.plist.PacketList:
        """Returns the packet list of captured packets."""
        return self._sniffer.results

    def pcap(self) -> bytes:
        """Returns the results in pcap format serialised to bytes."""
        return PacketSniffer.to_pcap(self.results)

    @staticmethod
    def to_pcap(packet_list: scapy.plist.PacketList) -> bytes:
        """Encodes the provided packet list in PCAP format."""
        byte_buffer = io.BytesIO()
        with scapy.utils.PcapWriter(byte_buffer) as writer:
            writer.write(packet_list)
            writer.flush()
            return byte_buffer.getvalue()

    def start(self) -> None:
        """Start capturing packets."""
        self._sniffer.start()
        self._logger.info('Waiting %fs for sniffer to initialise',
                          self.start_delay)
        time.sleep(self.start_delay)
        self._logger.info('Began sniffing for traffic with filter "%s"',
                          self._filter)

    def stop(self) -> None:
        """Stop capturing packets."""
        self._sniffer.stop()
        if self.results:
            self._logger.info('Sniffing complete. Captured %d packets',
                              len(self.results))
        else:
            self._logger.warning('Sniffing complete but failed to capture '
                                 'packets [result: %s]', self.results)


Result = TypedDict('Result', {
    'domain': Domain,
    'with_quic': bool,
    'page_source': Optional[str],
    'status': Literal['success', 'timeout', 'failure'],
    'http_trace': Dict[str, Any],
    'packets': scapy.plist.PacketList,
})


class WebsiteTraceExperiment:
    """Experiment consisting for fetching and tracing website traffic."""
    def __init__(self, sniffer: PacketSniffer, session_factory: SessionFactory):
        self._logger = logging.getLogger(__name__)
        self._sniffer = sniffer
        self._session_factory = session_factory

    def sample_domain(
        self, domain: Domain, repetitions: int = 1, stop_on_error: bool = True,
        keep_sources: Literal['all', 'first', 'none'] = 'all'
    ) -> SimpleGenerator[Result]:
        """Yields the results of sampling with and without QUIC.

        When save_all_sources is False, only the source of the first repetition
        will be provided.
        """
        failed = ''

        for repetition in range(1, repetitions + 1):
            with_source = keep_sources == 'all' or (
                keep_sources == 'first' and repetition == 1)

            quic_sample = self.sample(
                domain, use_quic=True, with_source=with_source)
            failed += '' if quic_sample['status'] == 'success' else 'QUIC'
            yield quic_sample

            tcp_sample = self.sample(
                domain, use_quic=False, with_source=with_source)
            failed += '' if tcp_sample['status'] == 'success' else (
                ' and TCP' if failed is not None else 'TCP')
            yield tcp_sample

            if failed and stop_on_error:
                self._logger.warning(
                    'Stopped at repetition %d for domain %s as %s failed.',
                    repetition, domain, failed)
                return

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
                self._logger.warning('Failed to fetch domain %s (quic: %s). %s',
                                     domain, use_quic, error)
                status = 'timeout'
            except (FetchFailed, UnexpectedAlertPresentException) as error:
                self._logger.warning('Failed to fetch domain %s (quic: %s). %s',
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
