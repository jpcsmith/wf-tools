"""Tests for the kfingerprinting.features module."""
# pylint: disable=redefined-outer-name,invalid-name
from pathlib import Path
from typing import Sequence
from itertools import product
from collections import namedtuple

import pytest
from pytest import approx
import numpy as np

from lab.trace import TraceData, Trace
from lab.classifiers.kfingerprinting import rf_fextract
from lab.classifiers.kfingerprinting import _features


def make_trace(direction: str, format_: str):
    """Return a trace with the specified direction and format."""
    assert direction in ("in", "out", "both"), "unrecognised direction"
    assert format_ in ("Trace", "recarray")

    path = Path(__file__).with_name('sample-trace.json')
    trace = TraceData.deserialise(path.read_text()).trace

    if direction in ("in", "out"):
        # Filter to packets in the single direction
        dir_value = 1 if direction == "out" else -1
        trace = [pkt for pkt in trace if pkt.direction == dir_value]

        # Adjust starting timestamp
        start = trace[0].timestamp
        trace = [pkt._replace(timestamp=(pkt.timestamp-start)) for pkt in trace]

    if format_ == "recarray":
        return np.asarray(trace, dtype=[
            ("timestamp", "f8"), ("direction", "i1"), ("size", "i8")
        ]).view(np.recarray), trace
    assert format_ == "Trace"
    return trace, trace


def trace_id(param: tuple) -> str:
    """Return a string to include in the test name for the given param of
    direction and format.
    """
    return f"direction={param[0]},format={param[1]}"


@pytest.fixture
def sample_trace() -> Trace:
    """Return a sample real world trace."""
    return make_trace("both", "Trace")[0]


@pytest.fixture
def split_trace(sample_trace) -> tuple:
    """Return a sample trace as timestamps and sizes."""
    trace = np.asarray(sample_trace)
    times = trace[:, 0]
    sizes = trace[:, 1] * trace[:, 2]
    return namedtuple(  # type: ignore
        "split_trace", ["timestamps", "sizes"])(times, sizes)


@pytest.fixture(
    name="trace", params=["recarray", "Trace"],
    ids=lambda x: trace_id(("both", x))
)
def fixture_trace(request):
    """Return trace data as a recarray or Trace. To get the same data
    as a trace use the sample_trace fixture.
    """
    return make_trace("both", format_=request.param)[0]


@pytest.fixture(
    params=product(["in", "out"], ["recarray", "Trace"]), ids=trace_id
)
def unidir_trace(request):
    """Return a unidirectional trace generated from the sample trace."""
    direction, format_ = request.param
    return make_trace(direction, format_)[0]


@pytest.fixture(
    params=product(["both", "in", "out"], ["recarray", "Trace"]), ids=trace_id
)
def trace_info(request) -> tuple:
    """Return a uni or bidirectional trace generated from the sample trace
    in recarray or Trace format, as well as in Trace format.
    """
    direction, format_ = request.param
    return make_trace(direction, format_)


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


def test_interarrival_stats_unidir(unidir_trace: Trace):
    """Test interarrival stats for unidirectional traces."""
    # Determine whether this trace is only outgoing or incoming packets
    is_outgoing = (unidir_trace[0].direction == 1)

    # Reference results taken from the same trace with both directions
    ref_result = {
        "interarrival::in::max": 0.44387899999999997,
        "interarrival::out::max": 0.42554800000000004,
        "interarrival::overall::max": 0.42554800000000004,
        "interarrival::in::mean": 0.009935120910384067,
        "interarrival::out::mean": 0.008797006281407036,
        "interarrival::overall::mean": 0.0046682780000000005,
        "interarrival::in::std": 0.0374760790985192,
        "interarrival::out::std": 0.033498408529192156,
        "interarrival::overall::std": 0.023948916296053623,
        "interarrival::in::percentile-75": 0.0030320000000000347,
        "interarrival::out::percentile-75": 0.001746500000000012,
        "interarrival::overall::percentile-75": 0.0005540000000000267,
    }
    for stat in ["max", "mean", "std", "percentile-75"]:
        # Set the stat of the other direction to NaN
        other_direction = "in" if is_outgoing else "out"
        ref_result[f"interarrival::{other_direction}::{stat}"] = np.nan
        # The overall stat is the same as whichever direction is present
        direction = "out" if is_outgoing else "in"
        ref_result[f"interarrival::overall::{stat}"] = \
            ref_result[f"interarrival::{direction}::{stat}"]

    result = _features.interarrival_stats(unidir_trace, allow_unidir=True)

    assert result == approx(ref_result, nan_ok=True)


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


def test_time_percentiles_unidir(unidir_trace: Trace):
    """The percentiles of the timestamps should allow unidirectional traces."""
    # Determine whether this trace is only outgoing or incoming packets
    is_outgoing = (unidir_trace[0].direction == 1)

    # Reference results taken from the output of the run in the unidirectional,
    # as the multidirectional setting does not account for the shift in
    # timestamps in the unidir setting.
    ref_result = {
        "time::in::percentile-25": 0.8506475,
        "time::in::percentile-50": 1.76016,
        "time::in::percentile-75": 4.02681875,
        "time::in::percentile-100": 6.98439,
        "time::out::percentile-25": 0.847308,
        "time::out::percentile-50": 1.301309,
        "time::out::percentile-75": 3.574881,
        "time::out::percentile-100": 7.002417,
    }

    for percentile in [25, 50, 75, 100]:
        # Set the stat of the other direction to NaN
        other_direction = "in" if is_outgoing else "out"
        ref_result[f"time::{other_direction}::percentile-{percentile}"] = np.nan
        # The overall stat is the same as whichever direction is present
        direction = "out" if is_outgoing else "in"
        ref_result[f"time::overall::percentile-{percentile}"] = \
            ref_result[f"time::{direction}::percentile-{percentile}"]

    result = _features.time_percentiles(unidir_trace, allow_unidir=True)
    assert result == approx(ref_result, nan_ok=True)


def test_packet_counts(trace, sample_trace: Trace):
    """The packet counts should match the reference."""
    ref_result = rf_fextract.number_pkt_stats(as_lines(sample_trace))
    result = _features.packet_counts(trace)

    assert ref_result == tuple(result.values())


def test_packet_counts_unidir(unidir_trace: Trace):
    """The packet counts should be zero for missing direction."""
    # Determine whether this trace is only outgoing or incoming packets
    is_outgoing = (unidir_trace[0].direction == 1)

    # Reference results taken from the full trace with both directions
    ref_result = {
        "packet-counts::in": 0 if is_outgoing else 704,
        "packet-counts::out": 797 if is_outgoing else 0,
        "packet-counts::overall": 797 if is_outgoing else 704,
    }

    result = _features.packet_counts(unidir_trace)
    assert ref_result == result


def test_head_tail_concentration(trace_info):
    """The concentration of first and last 30 packets should match the
    reference implementation.
    """
    trace, sample_trace = trace_info

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


def test_packet_concentration_stats_unidir(unidir_trace: Trace):
    """Concentrations should match the reference implementation."""
    chunk_size = 20
    # Determine whether this trace is only outgoing or incoming packets
    is_outgoing = (unidir_trace[0].direction == 1)

    # Since every chunk will have only outgoing packets, we can generate what
    # would be observed below
    n_packets = len(unidir_trace)
    ref_conc = []
    for _ in range(chunk_size, n_packets, chunk_size):
        ref_conc.append(chunk_size)
    if n_packets % chunk_size != 0:
        ref_conc.append(n_packets % chunk_size)

    ref_stats = {
        'std::out': np.std(ref_conc) if is_outgoing else 0,
        'mean::out': np.mean(ref_conc) if is_outgoing else 0,
        'median::out': np.median(ref_conc) if is_outgoing else 0,
        # The last would possibly have less than chunk_size packets
        'min::out': ref_conc[-1] if is_outgoing else 0,
        # All except the last would have the max
        'max::out': ref_conc[0] if is_outgoing else 0,
    }
    ref_stats = {
        f"concentration-stats::{key}": val for key, val in ref_stats.items()
    }

    stats, conc = _features.packet_concentration_stats(unidir_trace, chunk_size)
    assert ref_stats == stats
    ref_conc = ref_conc if is_outgoing else [0] * len(ref_conc)
    assert ref_conc == conc


def test_packets_per_second(trace_info):
    """The packets per second should match the reference implementation."""
    trace, sample_trace = trace_info

    *ref_stats, ref_pps = rf_fextract.number_per_sec(as_lines(sample_trace))
    stats, pps = _features.packets_per_second_stats(trace)

    assert ref_stats == list(stats.values())
    assert np.array_equal(ref_pps, pps)


def test_packet_ordering_stats(trace, sample_trace: Trace):
    """The packet ordering statistics should match the reference."""
    ref_stats = rf_fextract.avg_pkt_ordering_stats(as_lines(sample_trace))
    stats = _features.packet_ordering_stats(trace)

    assert ref_stats == tuple(stats.values())


def test_packet_ordering_stats_unidir(unidir_trace: Trace):
    """The packet ordering statistics should match the reference."""
    # Determine whether this trace is only outgoing or incoming packets
    is_outgoing = (unidir_trace[0].direction == 1)

    # Based on the calculation of the feature, it reduces to the below in the
    # case when there is no interleaving
    mean = np.mean(range(len(unidir_trace)))
    std = np.std(range(len(unidir_trace)))

    ref_stats = {
        "packet-order::out::mean": mean if is_outgoing else np.nan,
        "packet-order::in::mean": np.nan if is_outgoing else mean,
        "packet-order::out::std": std if is_outgoing else np.nan,
        "packet-order::in::std": np.nan if is_outgoing else std,
    }

    stats = _features.packet_ordering_stats(unidir_trace)

    assert stats == approx(ref_stats, nan_ok=True)


def test_in_out_fraction(trace_info):
    """The fraction of incoming and outgoing packets should match the
    reference implementation.
    """
    trace, sample_trace = trace_info

    ref_in, ref_out = rf_fextract.perc_inc_out(as_lines(sample_trace))
    result = _features.in_out_fraction(trace)

    assert ref_in == result['fraction-incoming']
    assert ref_out == result['fraction-outgoing']


def test_total_packet_sizes(trace_info):
    """It should return the correct amount of packet sizes."""
    trace, sample_trace = trace_info

    # trace parameterised and may be the same as sample_trace, that's okay.
    # We're primarily testing the version of trace that's a recarray here
    expected = _features.total_packet_sizes(sample_trace, allow_unidir=True)
    result = _features.total_packet_sizes(trace, allow_unidir=True)
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


# @pytest.fixture(name="dataset_seq")
# def fixture_dataset_seq(dataset) -> tuple:
#     """Return multipe (times, sizes, expected_features) where each is
#     an ndarray storing multiple samples.
#     """
#     sizes, times, _ = dataset
#
#     features = np.ndarray((len(sizes), 165), dtype=float)
#     for i, (size_row, times_row) in enumerate(zip(sizes, times)):
#         features[i] = _features.extract_features(
#             timestamps=times_row, sizes=size_row)
#
#     return times, sizes, features


@pytest.mark.parametrize('n_features', [100, 300])
def test_extract_features_sequence(dataset, n_features):
    """It should correct extract features that are provided as a sequence.
    """
    sizes, times, _ = dataset
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, max_size=n_features)

    expected_features = np.asarray([
        _features.extract_features(
            sizes=size_row, timestamps=time_row, max_size=n_features)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)


@pytest.mark.parametrize('n_features', [100, 300])
def test_extract_features_sequence_mp(dataset, n_features):
    """It should correct extract features that are provided as a sequence.
    """
    sizes, times, _ = dataset
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, max_size=n_features, n_jobs=3)

    expected_features = np.asarray([
        _features.extract_features(
            sizes=size_row, timestamps=time_row, max_size=n_features)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)


@pytest.mark.parametrize('n_jobs', [2, None])
def test_extract_features_sequence_mp_jobs(dataset, n_jobs):
    """It should correct extract features that are provided as a sequence.
    """
    sizes, times, _ = dataset
    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, n_jobs=n_jobs)

    expected_features = np.asarray([
        _features.extract_features(sizes=size_row, timestamps=time_row)
        for size_row, time_row in zip(sizes, times)
    ])
    np.testing.assert_allclose(expected_features, features)


def test_extract_features_sequence_unidir(unidir_trace):
    """It should extract features for unidirectional traces."""
    times = [[pkt.timestamp for pkt in unidir_trace]]
    sizes = [[pkt.size for pkt in unidir_trace]]

    features = _features.extract_features_sequence(
        timestamps=times, sizes=sizes, allow_unidir=True
    )
    assert features.shape == (1, 165)
    assert np.any(np.isnan(features))
