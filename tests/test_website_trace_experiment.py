"""Tests for lab.fetch_websites.WebsiteTraceExperiment"""
# pylint: disable=redefined-outer-name
from typing import Iterable
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
        'capture-A', 'capture-B', 'capture-C', 'capture-D',
        'capture-E', 'capture-F', 'capture-G', 'capture-H'])
    return mock


@pytest.fixture
def session_factory():
    """Returns a factory for creating mock sessions."""
    return create_mock_factory({
        'page_source': f'source-{val}',
        'fetch_page.return_value': f'source-{val}',
        'performance_log.return_value': f'trace-{val}',
    } for val in ['A', 'B', 'C', 'D'])


@pytest.fixture
def failed_session_factory():
    """Returns a factory for creating mock sessions."""
    return create_mock_factory({
        'page_source': None,
        'fetch_page.side_effect': FetchFailed,
        'performance_log.return_value': f'trace-{val}',
    } for val in ['A', 'B', 'C', 'D'])


@pytest.fixture
def seq_failed_session_factory():
    """Returns a factory for creating mock sessions which fail after 2
    successes.
    """
    return create_mock_factory([{
        'page_source': f'source-{val}',
        'fetch_page.return_value': f'source-{val}',
        'performance_log.return_value': f'trace-{val}',
    } for val in ['A', 'B']] + [{
        'page_source': None,
        'fetch_page.side_effect': FetchFailed,
        'performance_log.return_value': f'trace-{val}',
    } for val in ['C', 'D', 'E', 'F', 'G', 'H', 'I']])


def create_mock_factory(behaviours: Iterable[dict]):
    """Create a mock SessionFactory with sessions configured according to
    the provided behaviours.
    """
    mock_factory = Mock(spec=SessionFactory)
    mock_sessions = [
        MagicMock(spec=ChromiumSession, **behaviour) for behaviour in behaviours
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
        {'domain': domain, 'with_quic': True, 'page_source': 'source-B',
         'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
        {'domain': domain, 'with_quic': False, 'page_source': 'source-C',
         'status': 'success', 'http_trace': 'trace-C', 'packets': 'capture-C'},
        {'domain': domain, 'with_quic': False, 'page_source': 'source-D',
         'status': 'success', 'http_trace': 'trace-D', 'packets': 'capture-D'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, session_factory)

    result = list(experiment.sample_domain(domain, repetitions=2))

    assert result == expected


def test_initial_failure(sniffer, failed_session_factory):
    """It should stop the repetitions on an initial failure."""
    domain = Domain('example.com')
    expected = [
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-B', 'packets': 'capture-B'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, failed_session_factory)

    result = list(experiment.sample_domain(domain, repetitions=2))

    assert result == expected


def test_repeated_failure(sniffer, seq_failed_session_factory):
    """It should stop the repetitions on an repeated failures."""
    domain = Domain('example.com')
    expected = [
        # QUIC successes
        {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
         'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': True, 'page_source': 'source-B',
         'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
        # QUIC failures
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-C', 'packets': 'capture-C'},
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-D', 'packets': 'capture-D'},
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-E', 'packets': 'capture-E'},
        # TCP failure
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-F', 'packets': 'capture-F'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, seq_failed_session_factory)

    result = list(experiment.sample_domain(domain, repetitions=10))

    assert result == expected


def test_repeated_failure_continue(sniffer, seq_failed_session_factory):
    """It should stop the repetitions on an repeated failures."""
    domain = Domain('example.com')
    expected = [
        # QUIC successes
        {'domain': domain, 'with_quic': True, 'page_source': 'source-A',
         'status': 'success', 'http_trace': 'trace-A', 'packets': 'capture-A'},
        {'domain': domain, 'with_quic': True, 'page_source': 'source-B',
         'status': 'success', 'http_trace': 'trace-B', 'packets': 'capture-B'},
        # QUIC failures
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-C', 'packets': 'capture-C'},
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-D', 'packets': 'capture-D'},
        {'domain': domain, 'with_quic': True, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-E', 'packets': 'capture-E'},
        # TCP failure
        {'domain': domain, 'with_quic': False, 'page_source': None,
         'status': 'failure', 'http_trace': 'trace-F', 'packets': 'capture-F'},
    ]

    experiment = WebsiteTraceExperiment(sniffer, seq_failed_session_factory)

    result = list(experiment.sample_domain(domain, repetitions=10))

    assert result == expected
