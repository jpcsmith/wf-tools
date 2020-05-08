"""Tests for lab.fetch_websites.ChromiumSession."""
from unittest import mock
from unittest.mock import Mock, PropertyMock

import pytest
import selenium
from selenium.webdriver.remote.webdriver import WebDriver, WebDriverException

from lab.sniffer import PacketSniffer
from lab.fetch_websites import (
    ChromiumFactory, ChromiumSession, ChromiumSessionFactory, FetchTimeout,
    FetchFailed, Result, collect_trace
)


def test_begin_and_close_session(mocker):
    """The session should be start and stoppable, creating and quitting
    the driver on end.
    """
    mock_create = mocker.patch.object(ChromiumFactory, 'create', autospec=True)

    factory = ChromiumFactory()
    session = ChromiumSession(
        url="https://mall.com", protocol="h3-Q050", driver_factory=factory)

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
