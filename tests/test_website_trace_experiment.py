"""Tests for lab.fetch_websites.WebsiteTraceExperiment"""
# pylint: disable=redefined-outer-name
from unittest.mock import (
    Mock,
    MagicMock,
    PropertyMock,
)

import pytest

from lab.fetch_websites import (
    ChromiumSession,
    Domain,
    PacketSniffer,
    SessionFactory,
    WebsiteTraceExperiment,
    FetchFailed,
)


@pytest.fixture
def sniffer():
    """Returns a mocked sniffer."""
    mock = Mock(spec=PacketSniffer)
    type(mock).results = PropertyMock(side_effect=[
        'capture-A', 'capture-B', 'capture-C', 'capture-D'])
    return mock


@pytest.fixture
def session_factory():
    """Returns a factory for creating mock sessions."""
    mock_factory = Mock(spec=SessionFactory)
    mock_sessions = [
        MagicMock(spec=ChromiumSession, **{
            'page_source': f'source-{val}',
            'fetch_page.return_value': f'source-{val}',
            'performance_log.return_value': f'trace-{val}',
        }) for val in ['A', 'B', 'C', 'D']
    ]
    for mock in mock_sessions:
        mock.__enter__.return_value = mock

    mock_factory.create.side_effect = mock_sessions
    return mock_factory


@pytest.fixture
def failed_session_factory():
    """Returns a factory for creating mock sessions."""
    mock_factory = Mock(spec=SessionFactory)
    mock_sessions = [
        MagicMock(spec=ChromiumSession, **{
            'page_source': None,
            'fetch_page.side_effect': FetchFailed,
            'performance_log.return_value': f'trace-{val}',
        }) for val in ['A', 'B', 'C', 'D']
    ]
    for mock in mock_sessions:
        mock.__enter__.return_value = mock

    mock_factory.create.side_effect = mock_sessions
    return mock_factory


def test_sample_domain(sniffer, session_factory):
    """It should yield the samples for QUIC & TCP for domain."""
    domain = Domain('example.com')
    expected = [
        {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
         'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': False, 'page_source': 'source-B',
         'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
        {'domain': domain, 'with_quic': True, 'page_source': 'source-C',
         'status': 'success', 'http_trace': 'trace-C', 'packets': 'capture-C'},
        {'domain': domain, 'with_quic': False, 'page_source': 'source-D',
         'status': 'success', 'http_trace': 'trace-D', 'packets': 'capture-D'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, session_factory)

    result = list(experiment.sample_domain(domain, repetitions=2))

    assert result == expected


def test_no_stop_on_failure(sniffer, failed_session_factory):
    """It should continue the repetitions even on failure."""
    domain = Domain('example.com')
    expected = [
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-B', 'packets': 'capture-B'},
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-C', 'packets': 'capture-C'},
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-D', 'packets': 'capture-D'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, failed_session_factory)

    result = list(experiment.sample_domain(domain, repetitions=2,
                                           stop_on_error=False))

    assert result == expected


def test_stop_on_failure(sniffer, failed_session_factory):
    """It should stop the repetitions on failure."""
    domain = Domain('example.com')
    expected = [
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-B', 'packets': 'capture-B'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, failed_session_factory)

    result = list(experiment.sample_domain(domain, repetitions=2,
                                           stop_on_error=True))

    assert result == expected
