"""Tests for the kfingerprinting.features module."""
# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Sequence

import pytest

from lab.trace import TraceData, Trace
from lab.classifiers.kfingerprinting import rf_fextract
from lab.classifiers.kfingerprinting import _features


@pytest.fixture
def sample_trace() -> Trace:
    """Return a sample real world trace."""
    path = Path(__file__).with_name('sample-trace.json')
    return TraceData.deserialise(path.read_text()).trace


def as_lines(trace) -> Sequence[str]:
    """Return the trace as a sequence of lines."""
    return [' '.join(str(val) for val in pkt) for pkt in trace]


def test_interarrival_stats(sample_trace: Trace):
    """The interarrival stats computation should match the reference."""
    ref_result = rf_fextract.interarrival_maxminmeansd_stats(sample_trace)
    result = _features.interarrival_stats(sample_trace)

    assert ref_result == [(
        result['interarrival::in::max'], result['interarrival::out::max'],
        result['interarrival::overall::max'],
        result['interarrival::in::mean'], result['interarrival::out::mean'],
        result['interarrival::overall::mean'],
        result['interarrival::in::std'], result['interarrival::out::std'],
        result['interarrival::overall::std'],
        result['interarrival::in::percentile-75'],
        result['interarrival::out::percentile-75'],
        result['interarrival::overall::percentile-75'],
    )]


def test_time_percentiles(sample_trace: Trace):
    """The percentiles of the timestamps should match the reference."""
    ref_result = rf_fextract.time_percentile_stats(as_lines(sample_trace))
    result = _features.time_percentiles(sample_trace)

    assert ref_result == [
        result['time::in::percentile-25'], result['time::in::percentile-50'],
        result['time::in::percentile-75'], result['time::in::percentile-100'],
        result['time::out::percentile-25'], result['time::out::percentile-50'],
        result['time::out::percentile-75'], result['time::out::percentile-100'],
        result['time::overall::percentile-25'],
        result['time::overall::percentile-50'],
        result['time::overall::percentile-75'],
        result['time::overall::percentile-100'],
    ]


def test_packet_counts(sample_trace: Trace):
    """The packet counts should match the reference."""
    ref_result = rf_fextract.number_pkt_stats(as_lines(sample_trace))
    result = _features.packet_counts(sample_trace)

    assert ref_result == tuple(result.values())


def test_head_tail_concentration(sample_trace: Trace):
    """The concentration of first and last 30 packets should match the
    reference implementation.
    """
    ref_result = rf_fextract.first_and_last_30_pkts_stats(
        as_lines(sample_trace))
    result = _features.head_and_tail_concentration(sample_trace, 30)

    assert ref_result == list(result.values())


def test_packet_concentration_stats(sample_trace: Trace):
    """Concentrations should match the reference implementation."""
    *ref_stats, ref_conc = rf_fextract.pkt_concentration_stats(
        as_lines(sample_trace))
    stats, conc = _features.packet_concentration_stats(sample_trace, 20)

    assert ref_stats == list(stats.values())
    assert ref_conc == conc


def test_packets_per_second(sample_trace: Trace):
    """The packets per second should match the reference implementation."""
    *ref_stats, ref_pps = rf_fextract.number_per_sec(as_lines(sample_trace))
    stats, pps = _features.packets_per_second_stats(sample_trace)

    assert ref_stats == list(stats.values())
    assert ref_pps == pps


def test_packet_ordering_stats(sample_trace: Trace):
    """The packet ordering statistics should match the reference."""
    ref_stats = rf_fextract.avg_pkt_ordering_stats(as_lines(sample_trace))
    stats = _features.packet_ordering_stats(sample_trace)

    assert ref_stats == tuple(stats.values())


def test_in_out_fraction(sample_trace: Trace):
    """The fraction of incoming and outgoing packets should match the
    reference implementation.
    """
    ref_in, ref_out = rf_fextract.perc_inc_out(as_lines(sample_trace))
    result = _features.in_out_fraction(sample_trace)

    assert ref_in == result['fraction-incoming']
    assert ref_out == result['fraction-outgoing']


def test_extract_features(sample_trace: Trace):
    """The resulting feature sets should be equivalent for the
    first 46 features.
    """
    ref_features = rf_fextract.TOTAL_FEATURES(as_lines(sample_trace))
    features = _features.extract_features(sample_trace)

    assert ref_features[:46] == tuple(features[:46])


@pytest.mark.parametrize('n_features', [50, 100, 150, 200, 300])
def test_extract_features_length(n_features, sample_trace: Trace):
    """The number of features should be as expected."""
    features = _features.extract_features(sample_trace, n_features)
    assert len(features) == n_features
