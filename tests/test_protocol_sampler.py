"""Tests for lab.fetch_websites.ProtocolSampler"""
# pylint: disable=invalid-name
import asyncio
from unittest import mock
from unittest.mock import sentinel

import pytest
from mock.mock import AsyncMock

from lab.fetch_websites import ProtocolSampler


@pytest.mark.asyncio
async def test_sample_url_tries_each(mocker):
    """It should try that each protocol is successful before collecting a
    protocol repeatedly.
    """
    mock_collect_trace = mocker.patch(
        'lab.fetch_websites.collect_trace', autospec=True,
        side_effect=lambda u, proto, *_: dict(protocol=proto, status='success'))

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 2})
    _ = [result async for result in results]

    mock_collect_trace.assert_has_calls([
        mock.call('https://pie.ch', 'tcp', sentinel.sniffer, sentinel.factory),
        mock.call('https://pie.ch', 'Q043', sentinel.sniffer, sentinel.factory),
        mock.call('https://pie.ch', 'Q046', sentinel.sniffer, sentinel.factory),
    ], any_order=True)


@pytest.mark.asyncio
async def test_samples_repeatedly(mocker):
    """It should collect the required traces per protocol.
    """
    mock_collect_trace = mocker.patch(
        'lab.fetch_websites.collect_trace', autospec=True,
        side_effect=lambda u, proto, *_: dict(protocol=proto, status='success'))

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1})
    _ = [result async for result in results]

    mock_collect_trace.assert_has_calls([
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043']*2 + ['tcp']*3 + ['Q046']
    ], any_order=True)


@pytest.mark.asyncio
async def test_retries_on_failure(mocker):
    """Should retry failed attempts."""
    mock_collect_trace = mocker.patch(
        'lab.fetch_websites.collect_trace', autospec=True)
    mock_collect_trace.side_effect = [
        {'protocol': proto, 'status': status}
        for proto, status in [
            ('Q043', 'success'), ('tcp', 'failure'), ('tcp', 'success'),
            ('Q046', 'success'), ('Q043', 'success')
        ] + [('tcp', 'success')] * 2
    ]

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1})
    items = [(result['protocol'], result['status']) async for result in results]

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


@pytest.mark.asyncio
async def test_sample_url_max_attempts(mocker):
    """It should observe the max attempts, for sequential failures."""
    mock_collect_trace = mocker.patch(
        'lab.fetch_websites.collect_trace', autospec=True)
    mock_collect_trace.side_effect = [
        {'protocol': proto, 'status': status}
        for proto, status in [
            ('Q043', 'success'), ('tcp', 'timeout'), ('tcp', 'success'),
            ('Q046', 'failure'), ('Q046', 'success'), ('tcp', 'timeout'),
            ('tcp', 'failure'),
        ]
    ]

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
        max_attempts=2
    ).sample_url('https://pie.ch', {'Q043': 1, 'tcp': 5, 'Q046': 1})
    _ = [result async for result in results]

    assert mock_collect_trace.call_args_list == [
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043', 'tcp', 'tcp'] + ['Q046']*2 + ['tcp']*2]


@pytest.mark.asyncio
async def test_sample_url_max_attempts_per_protocol(mocker):
    """It should observe the max attempts, for sequential failures of a
    individual protocols.
    """
    statuses = {'quic': 'failure', 'tcp': 'success', 'Q043': 'failure',
                'Q046': 'success'}

    def _fail_single_protocols(_, protocol, *_args, **_kwargs):
        return {'protocol': protocol, 'status': statuses[protocol]}

    mock_collect_trace = mocker.patch(
        'lab.fetch_websites.collect_trace', autospec=True)
    mock_collect_trace.side_effect = _fail_single_protocols

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
        max_attempts=2, attempts_per_protocol=True,
    ).sample_url('https://pie.ch', {'tcp': 3, 'Q043': 3, 'Q046': 3, 'quic': 3})
    traces = [result async for result in results]

    assert traces == [
        {'protocol': proto, 'status': status} for proto, status in [
            ('tcp', 'success'), ('Q043', 'failure'), ('Q043', 'failure'),
            ('Q046', 'success'), ('quic', 'failure'), ('quic', 'failure'),
            ('tcp', 'success'), ('Q046', 'success'), ('tcp', 'success'),
            ('Q046', 'success')
        ]
    ]


@pytest.mark.asyncio
async def test_sample_url_delay(mocker):
    """It should sleep between successive calls for the same website."""
    sampler = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
        max_attempts=2, delay=30)

    mock_sequence = AsyncMock(side_effect=[
        {'protocol': proto, 'status': 'success'}
        for proto in ['Q043', None, 'tcp', None, 'tcp']
    ])
    mocker.patch.object(sampler, 'collect_trace', mock_sequence)
    mocker.patch('asyncio.sleep', mock_sequence)

    results = sampler.sample_url('https://pie.ch', {'Q043': 1, 'tcp': 2})
    _ = [result async for result in results]

    assert mock_sequence.await_args_list == [
        mock.call('https://pie.ch', 'Q043'), mock.call(30),
        mock.call('https://pie.ch', 'tcp'), mock.call(30),
        mock.call('https://pie.ch', 'tcp'),
    ]


@pytest.mark.asyncio
async def test_sample_multiple(mocker):
    """It should interleave multiple samples."""
    sampler = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory, delay=0.001)
    protocols = {'tcp': 1, 'Q043': 1}

    mock_sequence = AsyncMock(side_effect=[
        {'protocol': proto, 'status': 'success'}
        for proto in ['tcp', 'tcp', 'Q043', 'Q043']
    ], spec=sampler.collect_trace, spec_set=True)
    mocker.patch.object(sampler, 'collect_trace', mock_sequence)
    mock_sleep = mocker.spy(asyncio, 'sleep')

    urls = {'https://example.com': protocols.copy(),
            'https://google.com': protocols.copy()}
    _ = [x async for x in sampler.sample_multiple(urls)]

    assert mock_sequence.await_args_list == [
        mock.call('https://example.com', 'tcp'),
        mock.call('https://google.com', 'tcp'),
        mock.call('https://example.com', 'Q043'),
        mock.call('https://google.com', 'Q043'),
    ]
    # Sleeps should only be called twice
    assert sum(mock.call(0.001) == c for c in mock_sleep.call_args_list) == 2
