"""Tests for lab.fetch_websites.ChromiumSession."""
# pylint: disable=redefined-outer-name
from unittest.mock import Mock

import pytest
import selenium
from selenium.webdriver.remote.webdriver import WebDriver

from lab.fetch_websites import (
    ChromiumSession,
    Domain,
    FetchFailed,
    FetchTimeout,
    WebDriverFactory,
)


@pytest.fixture
def mock_driver():
    """Returns a mock WebDriver."""
    return Mock(spec=WebDriver)


@pytest.fixture
def driver_factory(mock_driver):
    """Returns a mock WebDriverFactory which creates mock WebDrivers."""
    factory = Mock(spec=WebDriverFactory)
    factory.create.return_value = mock_driver
    return factory


@pytest.fixture(params=[True, False], ids=['QUIC', 'TCP'])
def chromium_session(driver_factory, request):
    """Returns a QUIC enabled ChromiumSession instance for the domain
    example.com with a mock WebDriverFactory.
    """
    return ChromiumSession(Domain('example.com'), request.param, driver_factory)


@pytest.fixture
def open_session(chromium_session):
    """Returns a chromium session which has already started."""
    chromium_session.begin()
    return chromium_session


def test_session_begin(chromium_session, driver_factory):
    """It should create a new driver in begin."""
    chromium_session.begin()

    if chromium_session.use_quic is True:
        driver_factory.create.assert_called_once_with(Domain('example.com'))
    else:
        driver_factory.create.assert_called_once_with(None)


def test_session_close(open_session, mock_driver):
    """The session should release the driver resources on close."""
    open_session.close()
    mock_driver.close.assert_called_once()


def test_fetch_page_timeout(open_session, mock_driver):
    """It should raise 'FetchTimeout' on a request timeout."""
    mock_driver.get.side_effect = selenium.common.exceptions.TimeoutException()

    with pytest.raises(FetchTimeout):
        open_session.fetch_page(timeout=10)

    mock_driver.set_page_load_timeout.assert_called_once_with(10)


def test_fetch_page_failed(open_session, mock_driver):
    """It should raise a FetchFailed exception if the result is an empty html
    page.
    """
    page_source = "<html><head></head><body></body></html>"
    mock_driver.page_source = page_source

    with pytest.raises(FetchFailed):
        open_session.fetch_page()


def test_fetch_page_success(open_session, mock_driver):
    """It should return the page source on a successful request."""
    page_source = "<html>This is the page source.</html>"
    mock_driver.page_source = page_source

    result = open_session.fetch_page()

    assert result == page_source
    mock_driver.get.assert_called_once_with('https://example.com')
