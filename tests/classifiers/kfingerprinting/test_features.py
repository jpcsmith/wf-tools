"""Tests for the kfingerprinting.features module."""
# pylint: disable=redefined-outer-name
from pathlib import Path
from typing import Sequence, Tuple
from collections import namedtuple

import pytest
import numpy as np

from lab.trace import TraceData, Trace
from lab.classifiers.kfingerprinting import rf_fextract
from lab.classifiers.kfingerprinting import _features


@pytest.fixture
def sample_trace() -> Trace:
    """Return a sample real world trace."""
    path = Path(__file__).with_name('sample-trace.json')
    return TraceData.deserialise(path.read_text()).trace


@pytest.fixture
def split_trace(sample_trace) -> tuple:
    """Return a sample trace as timestamps and sizes."""
    trace = np.asarray(sample_trace)
    times = trace[:, 0]
    sizes = trace[:, 1] * trace[:, 2]
    return namedtuple(  # type: ignore
        "split_trace", ["timestamps", "sizes"])(times, sizes)


@pytest.fixture(name="trace", params=["recarray", "Trace"])
def fixture_trace(request, sample_trace):
    """Return trace data as a recarray or Trace. To get the same data
    as a trace use the sample_trace fixture.
    """
    if request.param == "Trace":
        return sample_trace

    assert request.param == "recarray"
    trace = np.asarray(sample_trace, dtype=[
        ("timestamp", "f8"), ("direction", "i1"), ("size", "i8")
    ])
    return trace.view(np.recarray)


def as_lines(trace) -> Sequence[str]:
    """Return the trace as a sequence of lines."""
    return [' '.join(str(val) for val in pkt) for pkt in trace]


def test_interarrival_stats(trace, sample_trace: Trace):
    """The interarrival stats computation should match the reference."""
    ref_result = rf_fextract.interarrival_maxminmeansd_stats(sample_trace)
    result = _features.interarrival_stats(trace)

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


def test_time_percentiles(trace, sample_trace: Trace):
    """The percentiles of the timestamps should match the reference."""
    ref_result = rf_fextract.time_percentile_stats(as_lines(sample_trace))
    result = _features.time_percentiles(trace)

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


def test_packet_counts(trace, sample_trace: Trace):
    """The packet counts should match the reference."""
    ref_result = rf_fextract.number_pkt_stats(as_lines(sample_trace))
    result = _features.packet_counts(trace)

    assert ref_result == tuple(result.values())


def test_head_tail_concentration(trace, sample_trace: Trace):
    """The concentration of first and last 30 packets should match the
    reference implementation.
    """
    ref_result = rf_fextract.first_and_last_30_pkts_stats(
        as_lines(sample_trace))
    result = _features.head_and_tail_concentration(trace, 30)

    assert ref_result == list(result.values())


def test_packet_concentration_stats(trace, sample_trace: Trace):
    """Concentrations should match the reference implementation."""
    *ref_stats, ref_conc = rf_fextract.pkt_concentration_stats(
        as_lines(sample_trace))
    stats, conc = _features.packet_concentration_stats(trace, 20)

    assert ref_stats == list(stats.values())
    assert ref_conc == conc


def test_packets_per_second(trace, sample_trace: Trace):
    """The packets per second should match the reference implementation."""
    *ref_stats, ref_pps = rf_fextract.number_per_sec(as_lines(sample_trace))
    stats, pps = _features.packets_per_second_stats(trace)

    assert ref_stats == list(stats.values())
    assert ref_pps == pps


def test_packet_ordering_stats(trace, sample_trace: Trace):
    """The packet ordering statistics should match the reference."""
    ref_stats = rf_fextract.avg_pkt_ordering_stats(as_lines(sample_trace))
    stats = _features.packet_ordering_stats(trace)

    assert ref_stats == tuple(stats.values())


def test_in_out_fraction(trace, sample_trace: Trace):
    """The fraction of incoming and outgoing packets should match the
    reference implementation.
    """
    ref_in, ref_out = rf_fextract.perc_inc_out(as_lines(sample_trace))
    result = _features.in_out_fraction(trace)

    assert ref_in == result['fraction-incoming']
    assert ref_out == result['fraction-outgoing']


def test_total_packet_sizes(trace, sample_trace):
    """It should return the correct amount of packet sizes."""
    # trace parameterised and may be the same as sample_trace, that's okay.
    # We're primarily testing the version of trace that's a recarray here
    expected = _features.total_packet_sizes(sample_trace)
    result = _features.total_packet_sizes(trace)
    assert expected == result


def test_extract_features(trace, sample_trace: Trace):
    """The resulting feature sets should be equivalent for the
    first 46 features.
    """
    ref_features = rf_fextract.TOTAL_FEATURES(as_lines(sample_trace))
    features = _features.extract_features(trace)

    assert ref_features[:46] == tuple(features[:46])


def test_make_trace_array(split_trace, sample_trace):
    """It should correctly convert a split_trace to a trace."""
    expected = sample_trace
    converted = _features.make_trace_array(
        timestamps=split_trace.timestamps, sizes=split_trace.sizes
    ).astype('f,f,f').view('f').reshape((-1, 3))
    np.testing.assert_allclose(expected, converted)


def test_extract_features_split(sample_trace: Trace, split_trace):
    """The resulting feature sets should be equivalent for the
    first 46 features.
    """
    times, sizes = split_trace
    features = _features.extract_features(sample_trace)
    features_split = _features.extract_features(sizes=sizes, timestamps=times)

    assert len(features) == len(features_split)
    assert len(_features.ALL_DEFAULT_FEATURES) == len(features_split)

    for expected, result, tag in zip(
        features, features_split, _features.ALL_DEFAULT_FEATURES
    ):
        np.testing.assert_allclose(
            expected, result, err_msg=f"Result differs in feature {tag!r}")


@pytest.mark.parametrize('n_features', [50, 100, 150, 200, 300])
def test_extract_features_length(n_features, trace):
    """The number of features should be as expected."""
    features = _features.extract_features(trace, n_features)
    assert len(features) == n_features


@pytest.fixture(name="dataset_seq")
def fixture_dataset_seq(dataset) -> tuple:
    """Return multipe (times, sizes, expected_features) where each is
    an ndarray storing multiple samples.
    """
    sizes, times, classes = dataset
    return times[:10], sizes[:10]
    times = times[:10]

    features = np.ndarray((len(sizes), 165), dtype=float)
    for i, (size_row, times_row) in enumerate(zip(sizes, times)):
        features[i] = _features.extract_features(
            timestamps=times_row, sizes=size_row)

    return times, sizes, features


@pytest.mark.parametrize('n_features', [100, 300])
def test_extract_features_sequence(dataset_seq, n_features):
    """It should correct extract features that are provided as a sequence.
    """
    times, sizes = dataset_seq
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, max_size=n_features)

    expected_features = np.asarray([
        _features.extract_features(
            sizes=size_row, timestamps=time_row, max_size=n_features)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)


@pytest.mark.parametrize('n_features', [100, 300])
def test_extract_features_sequence_mp(dataset_seq, n_features):
    """It should correct extract features that are provided as a sequence.
    """
    times, sizes = dataset_seq
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, max_size=n_features, n_jobs=3)

    expected_features = np.asarray([
        _features.extract_features(
            sizes=size_row, timestamps=time_row, max_size=n_features)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)


@pytest.mark.parametrize('n_jobs', [2, None])
def test_extract_features_sequence_mp_jobs(dataset_seq, n_jobs):
    """It should correct extract features that are provided as a sequence.
    """
    times, sizes = dataset_seq
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, n_jobs=n_jobs)

    expected_features = np.asarray([
        _features.extract_features(sizes=size_row, timestamps=time_row)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)
