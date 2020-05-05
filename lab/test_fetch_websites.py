"""Tests for lab.fetch_websites.WebsiteTraceExperiment"""
# pylint: disable=redefined-outer-name
import itertools
from itertools import islice
from unittest import mock
from unittest.mock import Mock, patch, PropertyMock, sentinel

import pytest
import selenium
from selenium.webdriver.remote.webdriver import WebDriver, WebDriverException

import lab.fetch_websites
from lab.fetch_websites import (
    ChromiumSession, options_for_quic, ChromiumFactory, FetchFailed,
    FetchTimeout, ChromiumSessionFactory, Result, collect_trace, sample_url,
)

from lab.sniffer import PacketSniffer


def test_begin_and_close_session():
    """The session should be start and stoppable, creating and quitting
    the driver on end.
    """
    with patch.object(ChromiumFactory, 'create', autospec=True) as mock_create:
        factory = ChromiumFactory()
        session = ChromiumSession(url="https://mall.com", protocol="h3-Q050",
                                  driver_factory=factory)

        session.begin()
        mock_create.assert_called_once_with(factory, "https://mall.com",
                                            "h3-Q050")
        assert session.driver is mock_create.return_value

        session.close()
        assert mock_create.return_value is not None
        mock_create.return_value.quit.assert_called_once()
        assert session.driver is None


def test_close_gracefully():
    """Closing should not raise any errors."""
    session = ChromiumSession(url="https://mall.com", protocol="h3-Q050")
    session.driver = Mock(spec=WebDriver)
    assert session.driver is not None
    session.driver.quit.side_effect = WebDriverException("failed to close")

    session.close()


def test_factory_create_for_quic(monkeypatch):
    """It should create the driver with appropriate options for the protocol.
    """
    mock_init = Mock(spec=selenium.webdriver.Chrome, strict=True, name='Chrome')
    monkeypatch.setattr(lab.fetch_websites.webdriver, "Chrome", mock_init)

    factory = ChromiumFactory(driver_path="path")
    driver = factory.create("https://google.com", "Q043")

    mock_init.assert_called_once_with(executable_path="path", options=mock.ANY)
    assert driver == mock_init.return_value

    # pylint: disable=unsubscriptable-object
    cli_args = mock_init.call_args[1]['options'].arguments
    assert '--origin-to-force-quic-on=google.com:443' in cli_args
    assert '--quic-version=QUIC_VERSION_43' in cli_args


def test_factory_create_retries(monkeypatch):
    """It should repeatedly retry creation on failure."""
    mock_init = Mock(spec=selenium.webdriver.Chrome, strict=True, name='Chrome')
    mock_init.side_effect = [WebDriverException("1"), mock_init.return_value]
    monkeypatch.setattr(lab.fetch_websites.webdriver, "Chrome", mock_init)

    factory = ChromiumFactory(max_attempts=5, retry_delay=0)

    driver = factory.create("https://google.com", "Q043")
    assert mock_init.call_count == 2
    assert driver == mock_init.return_value


def test_factory_create_give_up(monkeypatch):
    """It should fail if the number of failures excedes retries."""
    mock_init = Mock(spec=selenium.webdriver.Chrome, strict=True, name='Chrome')
    mock_init.side_effect = [WebDriverException("1"), WebDriverException("2"),
                             WebDriverException("3"), mock_init.return_value]
    monkeypatch.setattr(lab.fetch_websites.webdriver, "Chrome", mock_init)

    factory = ChromiumFactory(max_attempts=3, retry_delay=0)

    with pytest.raises(WebDriverException):
        _ = factory.create("https://google.com", "Q043")
    assert mock_init.call_count == 3


def test_options_for_quic():
    """It should return the correct arguments to download QUIC."""
    assert options_for_quic("https://google.com", "Q043") == [
        "--origin-to-force-quic-on=google.com:443",
        "--quic-version=QUIC_VERSION_43"]
    assert options_for_quic("https://www.blogspot.com", "h3-Q050") == [
        "--origin-to-force-quic-on=www.blogspot.com:443",
        "--quic-version=h3-Q050"]


def test_options_for_tcp():
    """It should disable QUIC when the website it requested via TCP."""
    assert options_for_quic("https://facebook.com", "tcp") == ["--disable-quic"]


@pytest.fixture(name='chromium_session')
def chromium_session_fixture() -> ChromiumSession:
    """Return a chromium session with a mocked driver."""
    session = ChromiumSession(url="https://mall.com", protocol="h3-Q050")
    session.driver = Mock(spec=WebDriver, name='ChromiumSessionMock')
    session.driver.get_log.return_value = []  # type: ignore
    return session


def test_fetch_page_success(chromium_session):
    """Test simple success for fetching a page."""
    html_doc = ("<html><head></head><body>This is my HTML document. "
                "Interesting, I know, but really this is just for "
                "testing</body></html>")
    chromium_session.driver.page_source = html_doc

    assert chromium_session.fetch_page() == html_doc
    chromium_session.driver.get.assert_called_once_with("https://mall.com")


def test_fetch_page_timeout(chromium_session):
    """It should raise 'FetchTimeout' on a request timeout."""
    mock_driver = chromium_session.driver
    mock_driver.get.side_effect = selenium.common.exceptions.TimeoutException()

    with pytest.raises(FetchTimeout):
        chromium_session.fetch_page(timeout=10)

    mock_driver.set_page_load_timeout.assert_called_once_with(10)


def test_fetch_page_failed(chromium_session):
    """It should raise a FetchFailed exception if the result is an empty html
    page.
    """
    page_source = "<html><head></head><body></body></html>"
    mock_driver = chromium_session.driver
    mock_driver.page_source = page_source

    with pytest.raises(FetchFailed):
        chromium_session.fetch_page()


@pytest.fixture(name='mock_session_factory')
def mock_session_factory_fixture():
    """Return a mock session factory that creates a mock session.
    """
    mock_factory = mock.create_autospec(
        spec=ChromiumSessionFactory, spec_set=True, name='MockCSFactory',
        instance=True)
    mock_factory.create.return_value = mock.create_autospec(
        spec=ChromiumSession, spec_set=True, name='MockSession', instance=True)
    return mock_factory


@pytest.fixture(name='mock_sniffer')
def mock_sniffer_fixture():
    """Return a mock PacketSniffer"""
    return mock.create_autospec(
        spec=PacketSniffer, spec_set=True, name='MockSniffer', instance=True)


def test_collect_trace_success(mock_session_factory, mock_sniffer):
    """It should correctly return success results."""
    expected: Result = {
        'url': 'https://google.com', 'protocol': 'h3-Q050',
        'page_source': '<html><body>This is the page source</body></html>',
        'final_url': 'https://www.google.com', 'status': 'success',
        'http_trace': [{'entry': 'value'}, {'entry': 'value'}],
        'packets': b'Packet trace',
    }

    mock_session = mock_session_factory.create.return_value
    type(mock_session).current_url = PropertyMock(
        return_value=expected['final_url'])
    mock_session.fetch_page.return_value = expected['page_source']
    mock_session.performance_log.return_value = expected['http_trace']
    mock_session.__enter__.return_value = mock_session

    type(mock_sniffer).results = PropertyMock(return_value=expected['packets'])

    result = collect_trace(
        url='https://google.com', protocol='h3-Q050', sniffer=mock_sniffer,
        session_factory=mock_session_factory)

    mock_session_factory.create.assert_called_once_with(
        expected['url'], expected['protocol'])
    mock_session.performance_log.assert_called_once()
    mock_session.fetch_page.assert_called_once()
    mock_sniffer.start.assert_called_once()
    mock_sniffer.stop.assert_called_once()

    assert result == expected


@pytest.mark.parametrize('status,side_effect', [
    ('timeout', FetchTimeout('https://google.com', 99)),
    ('failure', FetchFailed('Failure!'))
])
def test_collect_trace_failures(mock_session_factory, mock_sniffer, status,
                                side_effect):
    """It should correctly report a timeout."""
    expected: Result = {
        'url': 'https://google.com', 'protocol': 'h3-Q050',
        'page_source': None, 'final_url': None, 'status': status,
        'http_trace': [], 'packets': b'',
    }

    mock_session = mock_session_factory.create.return_value
    mock_session.fetch_page.side_effect = side_effect
    mock_session.__enter__.return_value = mock_session

    result = collect_trace(
        url='https://google.com', protocol='h3-Q050', sniffer=mock_sniffer,
        session_factory=mock_session_factory)

    assert result == expected
    mock_session_factory.create.assert_called_once_with(expected['url'],
                                                        expected['protocol'])
    mock_session.fetch_page.assert_called_once()
    mock_sniffer.start.assert_called_once()
    mock_sniffer.stop.assert_called_once()


@mock.patch('lab.fetch_websites.collect_trace', autospec=True)
def test_sample_url_tries_each(mock_collect_trace):
    """It should try that each protocol is successful before collecting a
    protocol repeatedly.
    """
    mock_collect_trace.side_effect = lambda u, proto, *_: {
        'protocol': proto, 'status': 'success'}

    results = sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 2},
                         sentinel.sniffer, sentinel.factory)
    list(islice(results, 3))

    mock_collect_trace.assert_has_calls([
        mock.call('https://pie.ch', 'tcp', sentinel.sniffer, sentinel.factory),
        mock.call('https://pie.ch', 'Q043', sentinel.sniffer, sentinel.factory),
        mock.call('https://pie.ch', 'Q046', sentinel.sniffer, sentinel.factory),
    ], any_order=True)


@mock.patch('lab.fetch_websites.collect_trace', autospec=True)
def test_samples_repeatedly(mock_collect_trace):
    """It should collect the required traces per protocol.
    """
    mock_collect_trace.side_effect = lambda u, proto, *_: {
        'protocol': proto, 'status': 'success'}

    results = sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1},
                         sentinel.sniffer, sentinel.factory)
    list(results)

    mock_collect_trace.assert_has_calls([
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043']*2 + ['tcp']*3 + ['Q046']
    ], any_order=True)


@mock.patch('lab.fetch_websites.collect_trace', autospec=True)
def test_retries_on_failure(mock_collect_trace):
    """Should retry failed attempts."""
    tcp_failed = False

    def _collect_trace(url, proto, *_):
        nonlocal tcp_failed
        if not tcp_failed and proto == 'tcp':
            tcp_failed = True
            return {'protocol': proto, 'status': 'failure', 'url': url}
        return {'protocol': proto, 'status': 'success', 'url': url}
    mock_collect_trace.side_effect = _collect_trace

    results = sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1},
                         sentinel.sniffer, sentinel.factory)
    items = [(r['protocol'], r['status']) for r in results]

    # It should retry
    mock_collect_trace.assert_has_calls([
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043']*2 + ['tcp']*(3+1) + ['Q046']
    ], any_order=True)

    # The failed attempt should be reported
    assert sorted(items) == sorted(
        [('Q043', 'success')]*2 + [('tcp', 'success')]*3
        + [('tcp', 'failure'), ('Q046', 'success')]
    )


@mock.patch('lab.fetch_websites.collect_trace', autospec=True)
def test_sample_url_max_attempts(mock_collect_trace):
    """It should observe the max attempts, for sequential failures."""
    trace_results = iter([
        ('Q043', 'success'),
        ('tcp', 'timeout'), ('tcp', 'success'),
        ('Q046', 'failure'), ('Q046', 'success'),
        ('tcp', 'timeout'), ('tcp', 'failure'),
    ])

    def _collect_trace(url, proto, *_):
        protocol, status = next(trace_results)
        assert proto == protocol
        return {'protocol': proto, 'status': status, 'url': url}
    mock_collect_trace.side_effect = _collect_trace

    results = sample_url('https://pie.ch', {'Q043': 1, 'tcp': 5, 'Q046': 1},
                         sentinel.sniffer, sentinel.factory, max_attempts=2)
    results = list(results)

    assert mock_collect_trace.call_args_list == [
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043', 'tcp', 'tcp'] + ['Q046']*2 + ['tcp']*2]


# @pytest.fixture
# def mock_driver():
#     """Returns a mock WebDriver."""
#     return Mock(spec=WebDriver)


# @pytest.fixture
# def driver_factory(mock_driver):
#     """Returns a mock WebDriverFactory which creates mock WebDrivers."""
#     factory = Mock(spec=WebDriverFactory)
#     factory.create.return_value = mock_driver
#     return factory
#
#
# @pytest.fixture(params=[True, False], ids=['QUIC', 'TCP'])
# def chromium_session(driver_factory, request):
#     """Returns a QUIC enabled ChromiumSession instance for the domain
#     example.com with a mock WebDriverFactory.
#     """
#     return ChromiumSession(Domain('example.com'), request.param, driver_factory)
#
#
# @pytest.fixture
# def open_session(chromium_session):
#     """Returns a chromium session which has already started."""
#     chromium_session.begin()
#     return chromium_session
# pylint: disable=line-too-long
# @pytest.fixture
# def sniffer():
#     """Returns a mocked sniffer."""
#     mock = Mock(spec=PacketSniffer)
#     type(mock).results = PropertyMock(side_effect=[
#         'capture-A', 'capture-B', 'capture-C', 'capture-D',
#         'capture-E', 'capture-F', 'capture-G', 'capture-H'])
#     return mock
#
#
# @pytest.fixture
# def session_factory():
#     """Returns a factory for creating mock sessions."""
#     return create_mock_factory({
#         'page_source': f'source-{val}',
#         'fetch_page.return_value': f'source-{val}',
#         'performance_log.return_value': f'trace-{val}',
#     } for val in ['A', 'B', 'C', 'D'])
#
#
# @pytest.fixture
# def failed_session_factory():
#     """Returns a factory for creating mock sessions."""
#     return create_mock_factory({
#         'page_source': None,
#         'fetch_page.side_effect': FetchFailed,
#         'performance_log.return_value': f'trace-{val}',
#     } for val in ['A', 'B', 'C', 'D'])
#
#
# @pytest.fixture
# def seq_failed_session_factory():
#     """Returns a factory for creating mock sessions which fail after 2
#     successes.
#     """
#     return create_mock_factory([{
#         'page_source': f'source-{val}',
#         'fetch_page.return_value': f'source-{val}',
#         'performance_log.return_value': f'trace-{val}',
#     } for val in ['A', 'B']] + [{
#         'page_source': None,
#         'fetch_page.side_effect': FetchFailed,
#         'performance_log.return_value': f'trace-{val}',
#     } for val in ['C', 'D', 'E', 'F', 'G', 'H', 'I']])
#
#
# def create_mock_factory(behaviours: Iterable[dict]):
#     """Create a mock SessionFactory with sessions configured according to
#     the provided behaviours.
#     """
#     mock_factory = Mock(spec=SessionFactory)
#     mock_sessions = [
#         MagicMock(spec=ChromiumSession, **behaviour) for behaviour in behaviours
#     ]
#     for mock in mock_sessions:
#         mock.__enter__.return_value = mock
#
#     mock_factory.create.side_effect = mock_sessions
#     return mock_factory
#
#
# def test_sample_domain(sniffer, session_factory):
#     """It should yield the samples for QUIC & TCP for domain."""
#     domain = Domain('example.com')
#     expected = [
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
#          'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-B',
#          'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
#         {'domain': domain, 'with_quic': False, 'page_source': 'source-C',
#          'status': 'success', 'http_trace': 'trace-C', 'packets': 'capture-C'},
#         {'domain': domain, 'with_quic': False, 'page_source': 'source-D',
#          'status': 'success', 'http_trace': 'trace-D', 'packets': 'capture-D'},
#     ]
#
#     experiment = WebsiteTraceExperiment(sniffer, session_factory)
#
#     result = list(experiment.sample_domain(domain, repetitions=2))
#
#     assert result == expected
#
#
# def test_initial_failure(sniffer, failed_session_factory):
#     """It should stop the repetitions on an initial failure."""
#     domain = Domain('example.com')
#     expected = [
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-A', 'packets': 'capture-A'},
#         {'domain': domain, 'with_quic': False, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-B', 'packets': 'capture-B'},
#     ]
#
#     experiment = WebsiteTraceExperiment(sniffer, failed_session_factory)
#
#     result = list(experiment.sample_domain(domain, repetitions=2))
#
#     assert result == expected
#
#
# def test_repeated_failure(sniffer, seq_failed_session_factory):
#     """It should stop the repetitions on an repeated failures."""
#     domain = Domain('example.com')
#     expected = [
#         # QUIC successes
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
#          'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-B',
#          'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
#         # QUIC failures
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-C', 'packets': 'capture-C'},
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-D', 'packets': 'capture-D'},
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-E', 'packets': 'capture-E'},
#         # TCP failure
#         {'domain': domain, 'with_quic': False, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-F', 'packets': 'capture-F'},
#     ]
#
#     experiment = WebsiteTraceExperiment(sniffer, seq_failed_session_factory)
#
#     result = list(experiment.sample_domain(domain, repetitions=10))
#
#     assert result == expected
#
#
# def test_repeated_failure_continue(sniffer):
#     """It should stop the repetitions on an repeated failures."""
#     factory = create_mock_factory([
#         {'page_source': f'source-A', 'fetch_page.return_value': f'source-A',
#          'performance_log.return_value': f'trace-A'},
#         {'page_source': None, 'fetch_page.side_effect': FetchFailed,
#          'performance_log.return_value': f'trace-B'},
#         {'page_source': None, 'fetch_page.side_effect': FetchFailed,
#          'performance_log.return_value': f'trace-C'},
#         {'page_source': f'source-A', 'fetch_page.return_value': f'source-D',
#          'performance_log.return_value': f'trace-D'},
#         {'page_source': None, 'fetch_page.side_effect': FetchFailed,
#          'performance_log.return_value': f'trace-E'},
#         {'page_source': f'source-F', 'fetch_page.return_value': f'source-F',
#          'performance_log.return_value': f'trace-F'},
#         {'page_source': None, 'fetch_page.side_effect': FetchFailed,
#          'performance_log.return_value': f'trace-G'}])
#
#     domain = Domain('example.com')
#     expected = [
#         # QUIC success
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
#          'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
#         # QUIC failures
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-B', 'packets': 'capture-B'},
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-C', 'packets': 'capture-C'},
#         # QUIC success
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-D',
#          'status': 'success', 'http_trace': 'trace-D', 'packets': 'capture-D'},
#         # QUIC failure
#         {'domain': domain, 'with_quic': True, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-E', 'packets': 'capture-E'},
#         # QUIC success
#         {'domain': domain, 'with_quic': True, 'page_source': 'source-F',
#          'status': 'success', 'http_trace': 'trace-F', 'packets': 'capture-F'},
#         # TCP failure
#         {'domain': domain, 'with_quic': False, 'page_source': None,
#          'status': 'failure', 'http_trace': 'trace-G', 'packets': 'capture-G'},
#     ]
#
#     experiment = WebsiteTraceExperiment(sniffer, factory)
#
#     result = list(experiment.sample_domain(domain, repetitions=3))
#
#     assert result == expected
