"""Tests for lab.fetch_websites.WebsiteTraceExperiment"""
# pylint: disable=redefined-outer-name
import asyncio
from itertools import islice
from unittest import mock
from unittest.mock import patch, sentinel

import pytest
from mock.mock import AsyncMock

from lab.fetch_websites import ProtocolSampler


@pytest.fixture(name='sample_url')
def sample_url_fixture():
    """Return a function that can be used to synchronously sample
    protocols.
    """
    sampler = ProtocolSampler(sniffer=sentinel.sniffer,
                              session_factory=sentinel.factory)

    async def _gather(url, protocols):
        results = []
        async for result in sampler.async_sample_url(url, protocols):
            results.append(result)
        return results

    return lambda u, p: asyncio.run(_gather(u, p))


@mock.patch('lab.fetch_websites.collect_trace', autospec=True)
def test_sample_url_tries_each(mock_collect_trace):
    """It should try that each protocol is successful before collecting a
    protocol repeatedly.
    """
    mock_collect_trace.side_effect = lambda u, proto, *_: {
        'protocol': proto, 'status': 'success'}

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 2})
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

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1})
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

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory
    ).sample_url('https://pie.ch', {'Q043': 2, 'tcp': 3, 'Q046': 1})
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

    results = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
        max_attempts=2
    ).sample_url('https://pie.ch', {'Q043': 1, 'tcp': 5, 'Q046': 1})
    results = list(results)

    assert mock_collect_trace.call_args_list == [
        mock.call('https://pie.ch', proto, sentinel.sniffer, sentinel.factory)
        for proto in ['Q043', 'tcp', 'tcp'] + ['Q046']*2 + ['tcp']*2]


def test_sample_url_delay():
    """It should sleep between successive calls for the same website."""
    mock_sequence = AsyncMock()
    mock_sequence.side_effect = [
        {'protocol': proto, 'status': 'success'}
        for proto in ['Q043', None, 'tcp', None, 'tcp']
    ]

    sampler = ProtocolSampler(
        sniffer=sentinel.sniffer, session_factory=sentinel.factory,
        max_attempts=2, delay=30)
    with patch.object(sampler, 'collect_trace', mock_sequence), \
            patch('asyncio.sleep', mock_sequence):
        results = sampler.sample_url('https://pie.ch', {'Q043': 1, 'tcp': 2})
        results = list(results)

    assert mock_sequence.await_args_list == [
        mock.call('https://pie.ch', 'Q043'), mock.call(30),
        mock.call('https://pie.ch', 'tcp'), mock.call(30),
        mock.call('https://pie.ch', 'tcp'),
    ]


# @pytest.mark.asyncio
# async def test_sample_multiple():
#     sampler = ProtocolSampler(
#         sniffer=sentinel.sniffer, session_factory=sentinel.factory, delay=0.1)
#     protocols = {'tcp': 1, 'Q043': 1}
#
#     with mock.patch('lab.fetch_websites.collect_trace', autospec=True) as mock_collect_trace:
#         stream = aiostream.stream.merge(
#             sampler.async_sample_url('https://pie.ch', protocols),
#             sampler.async_sample_url('https://bin.ch', protocols),
#         )
#         print(await stream.list())
#
#         assert mock_collect_trace.call_args_list == [
#             mock.call('https://pie.ch', 'tcp', sentinel.sniffer, sentinel.factory),
#             mock.call('https://bin.ch', 'tcp', sentinel.sniffer, sentinel.factory),
#             mock.call('https://pie.ch', 'Q043', sentinel.sniffer, sentinel.factory),
#             mock.call('https://bin.ch', 'Q043', sentinel.sniffer, sentinel.factory),
#         ]
