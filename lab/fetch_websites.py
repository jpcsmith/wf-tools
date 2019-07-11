"""This module is responsible for fetching the webpages"""
# pylint: disable=too-few-public-methods
import io
import abc
from abc import abstractmethod
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    List,
    TypeVar,
    Optional,
)

from mypy_extensions import TypedDict
from typing_extensions import Literal
import scapy.sendrecv
import selenium
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver

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
        driver = webdriver.Chrome(
            executable_path=self.driver_path, options=options)
        self._logger.info('Chromedriver created from %s with arguments %s.',
                          self.driver_path, options.arguments)
        return driver

    @staticmethod
    def chrome_options(quic_domain: Optional[Domain])\
            -> webdriver.ChromeOptions:
        """Provide a set of options for the chrome webdriver."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')

        # Necessary when running as root
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

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

    Params:
        domain: The domain being requested in the session
        use_quic: Whether to use QUIC or TCP in the request
        driver_path: The local path to the chrome driver
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


class PacketSniffer:
    """Class for capturing network traffic."""
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
        byte_buffer = io.BytesIO()
        with scapy.utils.PcapWriter(byte_buffer) as writer:
            writer.write(self.results)
            writer.flush()
            return byte_buffer.getvalue()

    def start(self) -> None:
        """Start capturing packets."""
        self._sniffer.start()
        self._logger.info('Began sniffing for traffic with filter "%s"',
                          self._filter)

    def stop(self) -> None:
        """Stop capturing packets."""
        self._sniffer.stop()
        self._logger.info('Sniffing complete. Captured %d packets',
                          len(self.results))


Result = TypedDict('Result', {
    'domain': str,
    'with_quic': bool,
    'page_source': Optional[str],
    'status': Literal['success', 'timeout', 'failure'],
    'http_trace': Dict[str, Any],
    'packets': scapy.plist.PacketList,
})


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    driver_path: str
    repetitions: int

    def __post_init__(self) -> None:
        assert self.driver_path
        assert self.repetitions > 0


class _StopOnErrorException(Exception):
    """Raised due to early termination of a fetch loop."""
    def __init__(self, counter: int, *args):
        super().__init__(*args)
        self.counter = counter


class WebsiteTraceExperiment:
    """Experiment consisting for fetching and tracing website traffic."""
    def __init__(self, config: ExperimentConfig, domains: List[Domain]):
        assert domains
        self._logger = logging.getLogger(__name__)
        self._config = config
        self._domains = domains
        self._sniffer = PacketSniffer()

    def run(self) -> SimpleGenerator[Result]:
        """Runs the experiment and yields the results for each domain requested
        a number times with QUIC and TCP.
        """
        for domain in self._domains:
            yield from self._run_for_domain(domain)

    def _run_for_domain(self, domain) -> SimpleGenerator[Result]:
        try:
            yield from self.fetch_repeatedly(domain, use_quic=True,
                                             stop_on_error=True)
            self._logger.info("Fetched all %d results with QUIC for domain %s",
                              self._config.repetitions, domain)
        except _StopOnErrorException as err:
            self._logger.info("Fetched %d/%d results with QUIC for domain %s",
                              err.counter, self._config.repetitions, domain)
            yield self.fetch_single(domain, use_quic=False)
            self._logger.info("Fetched a single result with TCP for domain %s "
                              "as QUIC erred on first request.", domain)
        else:
            yield from self.fetch_repeatedly(domain, use_quic=False,
                                             stop_on_error=False)
            self._logger.info("Fetched all %d results with TCP for domain %s",
                              self._config.repetitions, domain)

    def fetch_repeatedly(
        self, domain: Domain, use_quic: bool, stop_on_error: bool = False
    ) -> SimpleGenerator[Result]:
        """Fetches the domain and yields the results, as often as specified by
        `config.repetitions`.

        If the fetch fails no exception will be raised but the status will
        record the type of failure. If `stop_on_error` is True, no more fetches
        will be made.
        """
        for i in range(1, self._config.repetitions + 1):
            result = self.fetch_single(domain, use_quic)
            yield result
            if result['status'] != 'success' and stop_on_error:
                _StopOnErrorException(i)

    def fetch_single(self, domain: Domain, use_quic: bool) -> Result:
        """Fetch the domain and returns the result."""
        page_source = None
        status = None

        with ChromiumSession(domain, use_quic, self._config.driver_path) \
                as session:
            self._sniffer.start()
            try:
                page_source = session.fetch_page()
                status = Literal['success']
            except FetchTimeout as error:
                self._logger.warning(
                    'Failed to fetch domain %s. %s', domain, error)
                page_source = session.page_source
                status = Literal['timeout']
            except FetchFailed as error:
                self._logger.warning(
                    'Failed to fetch domain %s. %s', domain, error)
                status = Literal['failure']

            return dict(page_source=page_source, status=status,
                        domain=domain.name, with_quic=use_quic,
                        http_trace=session.performance_log(),
                        packets=self._sniffer.results)


# def test_exp():
#     database = tinydb.TinyDB('db.json')
#     config = ExperimentConfig(
#         driver_path='tools/chromedriver',
#         repetitions=1)
#     experiment = WebsiteTraceExperiment(
#         config, ['example.com', 'mail.google.com'])
#     for result in experiment.run():
#         result['packets'] = base64.b64encode(
#             serialize_packets(result['packets'])).decode('utf8')
#         database.insert(result)
