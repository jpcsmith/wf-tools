"""Tests for lab.fetch_websites.ChromiumFactory."""
# pylint: disable=invalid-name
from unittest import mock
from unittest.mock import Mock

import pytest
import selenium
from selenium.webdriver.remote.webdriver import WebDriverException

import lab.fetch_websites
from lab.fetch_websites import options_for_quic, ChromiumFactory


def test_create_retries(monkeypatch):
    """It should repeatedly retry creation on failure."""
    mock_init = Mock(spec=selenium.webdriver.Chrome, strict=True, name='Chrome')
    mock_init.side_effect = [WebDriverException("1"), mock_init.return_value]
    monkeypatch.setattr(lab.fetch_websites.webdriver, "Chrome", mock_init)

    factory = ChromiumFactory(max_attempts=5, retry_delay=0)

    driver = factory.create("https://google.com", "Q043")
    assert mock_init.call_count == 2
    assert driver == mock_init.return_value


def test_create_give_up(monkeypatch):
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
        "--enable-quic",
        "--origin-to-force-quic-on=google.com:443",
        "--quic-version=QUIC_VERSION_43"]
    assert options_for_quic("https://www.blogspot.com", "h3-Q050") == [
        "--enable-quic",
        "--origin-to-force-quic-on=www.blogspot.com:443",
        "--quic-version=h3-Q050"]


def test_options_for_quic_multiple_version():
    """It should allow any quic version for protocol "QUIC" or "quic"."""
    assert options_for_quic("https://www.blogspot.com", "quic") == [
        "--enable-quic",
        "--origin-to-force-quic-on=www.blogspot.com:443"]
    assert options_for_quic("https://www.blogspot.com", "QUIC") == [
        "--enable-quic",
        "--origin-to-force-quic-on=www.blogspot.com:443"]


def test_options_for_tcp():
    """It should disable QUIC when the website it requested via TCP."""
    assert options_for_quic("https://facebook.com", "tcp") == ["--disable-quic"]


def test_create_for_quic(monkeypatch):
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


def test_create_for_quic_asterisk(monkeypatch):
    """It should create the driver with appropriate options for the protocol.
    """
    mock_init = Mock(spec=selenium.webdriver.Chrome, strict=True, name='Chrome')
    monkeypatch.setattr(lab.fetch_websites.webdriver, "Chrome", mock_init)

    factory = ChromiumFactory(driver_path="path", force_quic_on_all=True)
    driver = factory.create("https://google.com", "Q043")

    mock_init.assert_called_once_with(executable_path="path", options=mock.ANY)
    assert driver == mock_init.return_value

    # pylint: disable=unsubscriptable-object
    cli_args = mock_init.call_args[1]['options'].arguments
    assert '--origin-to-force-quic-on=google.com:443' not in cli_args
    assert '--origin-to-force-quic-on=*' in cli_args
    assert '--quic-version=QUIC_VERSION_43' in cli_args
