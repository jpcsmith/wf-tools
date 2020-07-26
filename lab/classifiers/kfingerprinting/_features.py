"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

Minor adjustments to the work of the original authors from the paper. The
original can be found at https://github.com/jhayes14/k-FP.
"""
import math
import logging
import itertools
import functools
import tempfile
from typing import Tuple, Union, Sequence, Optional
import multiprocessing

import h5py
import numpy as np

from lab.trace import Direction, Trace

DEFAULT_NUM_FEATURES = 165
_LOGGER = logging.getLogger(__name__)


# --------------------
# Non-feeder functions
# --------------------
def split_in_out(list_data: Trace, check: bool = True) -> Tuple[Trace, Trace]:
    """Returns a tuple of the packets in the (incoming, outgoing) subtraces.

    Raise AssertionError if check is true and the trace has no incoming or no
    outgoing packets.
    """
    # Use a fast-path for np record arrays
    if isinstance(list_data, np.recarray):
        incoming = list_data[list_data["direction"] < 0]
        outgoing = list_data[list_data["direction"] > 0]
    else:
        incoming = [pkt for pkt in list_data if pkt.direction == Direction.IN]
        outgoing = [pkt for pkt in list_data if pkt.direction == Direction.OUT]
    if check:
        assert len(incoming) > 0 and len(outgoing) > 0
    return (incoming, outgoing)


def _get_timestamps(array_like) -> np.ndarray:
    if isinstance(array_like, np.recarray):
        return array_like["timestamp"]
    return np.array([x[0] for x in array_like])


# -------------
# TIME FEATURES
# -------------
def _inter_pkt_time(list_data):
    if len(list_data) == 1:
        return [0.0, ]

    times = _get_timestamps(list_data)
    return (np.concatenate((times[1:], [times[0]])) - times)[:-1]


def interarrival_times(list_data):
    """Return the interarrival times of the incoming, outgoing, and overall
    packet sequences.
    """
    incoming, outgoing = split_in_out(list_data)
    inter_in = _inter_pkt_time(incoming)
    inter_out = _inter_pkt_time(outgoing)
    inter_overall = _inter_pkt_time(list_data)
    return inter_in, inter_out, inter_overall


def _prefix_keys(mapping: dict, prefix: Union[str, Sequence[str]]) -> dict:
    if not isinstance(prefix, str):
        prefix = '::'.join(prefix)
    return {f'{prefix}::{key}': mapping[key] for key in mapping}


def _interarrival_stats(times: Sequence[float]) -> dict:
    return {
        'mean': np.mean(times) if len(times) > 0 else 0,
        'max': max(times, default=0),
        'std': np.std(times) if len(times) > 0 else 0,
        'percentile-75': np.percentile(times, 75) if len(times) > 0 else 0
    }


def interarrival_stats(list_data: Trace) -> dict:
    """Extract the mean, std, max, 75th-percentile for the incoming,
    outgoing, and overall traces.
    """
    incoming, outgoing, overall = interarrival_times(list_data)
    return {
        **_prefix_keys(_interarrival_stats(incoming), ['interarrival', 'in']),
        **_prefix_keys(_interarrival_stats(outgoing), ['interarrival', 'out']),
        **_prefix_keys(_interarrival_stats(overall),
                       ['interarrival', 'overall']),
    }


def time_percentiles(overall: Trace) -> dict:
    """Return the 25th, 50th, 75th and 100th percentiles of the timestamps."""
    incoming, outgoing = split_in_out(overall)

    def _percentiles(trace):
        times = _get_timestamps(trace)
        return {f'percentile-{p}': (np.percentile(times, p)
                                    if len(times) > 0 else 0)
                for p in [25, 50, 75, 100]}

    return {
        **_prefix_keys(_percentiles(incoming), ['time', 'in']),
        **_prefix_keys(_percentiles(outgoing), ['time', 'out']),
        **_prefix_keys(_percentiles(overall), ['time', 'overall']),
    }


def packet_counts(overall: Trace) -> dict:
    """Return the number of incoming, outgoing and combined packets."""
    incoming, outgoing = split_in_out(overall, check=False)
    return {
        'packet-counts::in': len(incoming),
        'packet-counts::out': len(outgoing),
        'packet-counts::overall': len(overall)
    }


def head_and_tail_concentration(overall: Trace, count: int) -> dict:
    """Return the number of incoming and outgoing packets in the first and last
    'count' packets of the trace.
    """
    assert count > 0
    head = packet_counts(overall[:count])
    del head['packet-counts::overall']
    tail = packet_counts(overall[-count:])
    del tail['packet-counts::overall']

    return {
        **_prefix_keys(head, f'first-{count}'),
        **_prefix_keys(tail, f'last-{count}')
    }


def packet_concentration_stats(overall: Trace, chunk_size: int) \
        -> Tuple[dict, Sequence[int]]:
    """Return the std, mean, min, max and median of the number of
    outgoing packets in each chunk of the trace; as well as the
    sequence of outgoing concentrations.

    Each chunk is created with 'chunk_size' packets.
    """
    concentrations = []
    for index in range(0, len(overall), chunk_size):
        chunk = overall[index:(index + chunk_size)]
        concentrations.append(packet_counts(chunk)['packet-counts::out'])

    return _prefix_keys({
        'std::out': np.std(concentrations),
        'mean::out': np.mean(concentrations),
        'median::out': np.median(concentrations),
        'min::out': min(concentrations),
        'max::out': max(concentrations),
    }, 'concentration-stats'), concentrations


def alternate_concentration(concentration: Sequence[int], length: int) \
        -> Sequence[int]:
    """Return a fixed length sequence of the number of outgoing packets.

    The sequence of concentrations, where each value is the number of
    outgoing packets in a set of 20, is then partitioned into 20 sequences
    and each sequence is summed.  This roughly equates to divide the original
    sequence into 20 and counting the # of outgoing packets in each.  They
    differ as the resulting groups may slighly vary depending on the length
    of the sequence.  We therefore use the approach from the paper.
    """
    # We use the array_split implementation as the chunkIt code was flawed and
    # may return more chunks than requested.
    result = [sum(group) for group in np.array_split(concentration, length)]
    assert len(result) == length
    return result


def alternate_packets_per_second(pps: Sequence[int], length: int) \
        -> Tuple[dict, Sequence[int]]:
    """Return a fixed length sequence of the pps rate, as well as the sum of
    the rate
    """
    # We use the array_split implementation as the chunkIt code was flawed and
    # may return more chunks than requested.
    result = [sum(group) for group in np.array_split(pps, length)]
    assert len(result) == length
    return {'alt-pps::sum': sum(result)}, result


def packets_per_second_stats(overall: Trace) \
        -> Tuple[dict, Sequence[int]]:
    """Return the mean, std, min, median and max number of packets per
    second, as well as the number of packets each second.
    """
    n_seconds = math.ceil(overall[-1].timestamp)
    packets_per_sec, _ = np.histogram(
        _get_timestamps(overall), bins=n_seconds, range=(0, n_seconds))
    packets_per_sec = list(packets_per_sec)

    return {
        'pps::mean': np.mean(packets_per_sec),
        'pps::std': np.std(packets_per_sec),
        'pps::median': np.median(packets_per_sec),
        'pps::min': min(packets_per_sec),
        'pps::max': max(packets_per_sec)
    }, packets_per_sec


def packet_ordering_stats(overall: Trace) -> dict:
    """Mean and std of a variant of the packet ordering features."""
    # Note that the ordering here is different from the k-fingerprinting
    # reference implementation. They have out and in swapped.
    if isinstance(overall, np.recarray):
        in_preceeding = np.nonzero(overall["direction"] < 0)[0]
        out_preceeding = np.nonzero(overall["direction"] > 0)[0]
    else:
        in_preceeding = [i for i, pkt in enumerate(overall)
                         if pkt.direction == Direction.IN]
        out_preceeding = [i for i, pkt in enumerate(overall)
                          if pkt.direction == Direction.OUT]

    return {
        'packet-order::out::mean': np.mean(out_preceeding),
        'packet-order::in::mean': np.mean(in_preceeding),
        'packet-order::out::std': np.std(out_preceeding),
        'packet-order::in::std': np.std(in_preceeding),
    }


def in_out_fraction(overall: Trace) -> dict:
    """Return the fraction of incoming and outgoing packets."""
    counts = packet_counts(overall)
    n_packets = counts['packet-counts::overall']
    return {
        'fraction-incoming': counts['packet-counts::in'] / n_packets,
        'fraction-outgoing': counts['packet-counts::out'] / n_packets
    }


# -------------
# SIZE FEATURES
# -------------
def _get_sizes(array_like):
    if isinstance(array_like, np.recarray):
        return array_like["size"]
    return [x[2] for x in array_like]


def total_packet_sizes(overall: Trace) -> dict:
    """Return the total incoming, outgoing and overall packet sizes."""
    incoming, outgoing = split_in_out(overall)

    # Use absolute value in case the input sizes are signed
    result = {
        'total-size::in': np.sum(np.abs(_get_sizes(incoming))),
        'total-size::out': np.sum(np.abs(_get_sizes(outgoing))),
    }
    result['total-size::overall'] = result['total-size::in'] \
        + result['total-size::out']
    return result


def _packet_size_stats(trace: Trace) -> dict:
    sizes = _get_sizes(trace)
    return {
        'mean': np.mean(sizes), 'var': np.var(sizes),
        'std': np.std(sizes), 'max': np.max(sizes)
    }


def packet_size_stats(overall: Trace) -> dict:
    """Return the mean, var, std, and max of the incoming, outgoing,
    and overall packet traces.
    """
    incoming, outgoing = split_in_out(overall)
    return {
        **_prefix_keys(_packet_size_stats(incoming), 'size-stats::in'),
        **_prefix_keys(_packet_size_stats(outgoing), 'size-stats::out'),
        **_prefix_keys(_packet_size_stats(overall), 'size-stats::overall'),
    }


# ----------------
# FEATURE FUNCTION
# ----------------
def make_trace_array(
    timestamps: Sequence[float], sizes: Sequence[float]
) -> np.ndarray:
    """Create a trace-like array from the sequence of timestamps and
    signed sizes.
    """
    assert len(timestamps) == len(sizes)

    trace_array = np.recarray((len(timestamps), ), dtype=[
        # Use i8 for sizes since we may be doing operations which overflow
        ("timestamp", "f8"), ("direction", "i1"), ("size", "i8")
    ])
    trace_array["timestamp"] = timestamps

    sizes = np.asarray(sizes, dtype=int)
    np.sign(sizes, out=trace_array["direction"])
    np.abs(sizes, out=trace_array["size"])
    return trace_array


def _run_extraction(idx, directory: str, max_size: int):
    # Use copies so that the original memory of the full file may be freed
    with h5py.File(f"{directory}/data.hdf", mode="r") as h5file:
        sizes = np.asarray(h5file["sizes"][idx], dtype=np.object)
        times = np.asarray(h5file["timestamps"][idx], dtype=np.object)

    return _extract_features_local(
        timestamps=times, sizes=sizes, max_size=max_size)


def _extract_features_mp(
    timestamps: Sequence[Sequence[float]], sizes: Sequence[Sequence[float]],
    max_size: int = DEFAULT_NUM_FEATURES, n_jobs: Optional[int] = None
) -> np.ndarray:
    features = np.zeros((len(sizes), max_size), float)

    # Serialise the timestamps and sizes to file
    with tempfile.TemporaryDirectory(prefix="kfp-extract-") as directory:
        with h5py.File(f"{directory}/data.hdf", mode="w") as h5file:
            dtype = h5py.vlen_dtype(np.dtype("float"))
            h5file.create_dataset("sizes", data=sizes, dtype=dtype)
            h5file.create_dataset("timestamps", data=timestamps, dtype=dtype)

        offset = 0
        # Use our own splits as imap chunking would yield them one at a time
        chunksize = 5000
        n_chunks = max(len(sizes) // chunksize, 1)
        splits = np.array_split(np.arange(len(sizes)), n_chunks)
        assert n_chunks == len(splits)
        _LOGGER.info("Extracting features in %d batches...", n_chunks)

        with multiprocessing.Pool(n_jobs) as pool:
            # Pass the filenames and indices to the background process
            for i, batch in enumerate(pool.imap(
                functools.partial(
                    _run_extraction, directory=directory, max_size=max_size),
                splits, chunksize=1
            )):
                # Recombine them filenames and indices
                features[offset:offset+len(batch), :] = batch
                offset += len(batch)

                _LOGGER.info("Extraction is %.2f%% complete.",
                             ((i+1) * 100 / n_chunks))

    return features


def _extract_features_local(
    timestamps: Sequence[Sequence[float]], sizes: Sequence[Sequence[float]],
    max_size: int = DEFAULT_NUM_FEATURES
) -> np.ndarray:
    features = np.ndarray((len(sizes), max_size), dtype=float)

    for i, (size_row, times_row) in enumerate(zip(sizes, timestamps)):
        features[i] = extract_features(
            timestamps=times_row, sizes=size_row, max_size=max_size)

    return features


def extract_features_sequence(
    trace: Optional[Sequence[Trace]] = None,
    max_size: int = DEFAULT_NUM_FEATURES,
    timestamps: Optional[Sequence[Sequence[float]]] = None,
    sizes: Optional[Sequence[Sequence[float]]] = None,
    n_jobs: Optional[int] = 1
) -> np.ndarray:
    """Convenience method around extract_features that accepts a
    sequence of timestamps and sizes for multiple samples.

    If n_jobs is provided, use multiple processes to extract the
    features. An n_jobs of None will use all available processes
    """
    if trace is not None:
        raise NotImplementedError("Trace input not currently supported.")
    assert timestamps is not None
    assert sizes is not None

    if n_jobs != 1:
        _LOGGER.info("Extracting features using %r processes", n_jobs)
        return _extract_features_mp(
            timestamps=timestamps, sizes=sizes, max_size=max_size,
            n_jobs=n_jobs)

    _LOGGER.info("Extracting features locally.")
    return _extract_features_local(
        timestamps=timestamps, sizes=sizes, max_size=max_size)


def extract_features(
    trace: Trace = None, max_size: int = DEFAULT_NUM_FEATURES,
    timestamps: Optional[Sequence[float]] = None,
    sizes: Optional[Sequence[float]] = None
) -> np.ndarray:
    """Return a tuple of features of the specified size, according to the paper

        Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
        scalable website fingerprinting technique." 25th {USENIX} Security
        Symposium ({USENIX} Security 16). 2016.

    Either trace or both sizes and timestamps must be specified.
    """
    if trace is None and (timestamps is None or sizes is None):
        raise ValueError("timestamps and sizes must be specified when trace is "
                         "None.")
    if trace is not None and (timestamps is not None or sizes is not None):
        raise ValueError("Either trace or both sizes and timestamps should be "
                         "specified.")
    if trace is None:
        assert timestamps is not None and sizes is not None
        trace = make_trace_array(timestamps=timestamps, sizes=sizes)

    assert trace[0].timestamp == 0

    all_features = {}
    all_features.update(interarrival_stats(trace))
    all_features.update(time_percentiles(trace))
    all_features.update(packet_counts(trace))
    all_features.update(head_and_tail_concentration(trace, 30))

    stats, concentrations = packet_concentration_stats(trace, 20)
    all_features.update(stats)

    stats, pps = packets_per_second_stats(trace)
    all_features.update(stats)

    all_features.update(packet_ordering_stats(trace))
    all_features.update(in_out_fraction(trace))

    all_features.update(total_packet_sizes(trace))
    all_features.update(packet_size_stats(trace))

    result = [all_features[feat] for feat in DEFAULT_TIMING_FEATURES]

    # Alternate concentration feature
    result.extend(alternate_concentration(concentrations, 20))

    # Alternate packets per second features
    stats, alt_pps = alternate_packets_per_second(pps, 20)
    result.extend(alt_pps)
    result.append(stats['alt-pps::sum'])

    # Assert on the length of the core features from the paper
    assert len(result) == 87

    result.extend(all_features[feat] for feat in DEFAULT_SIZE_FEATURES)

    # Assert on the overall length of the sizes and timing features
    assert len(result) == 102

    remaining_space = max_size - len(result)

    # Align the concentrations and pps features, by allocating each roughly
    # Half of the remaining space, padding with zero otherwise
    if remaining_space > 0:
        _extend_exactly(result, concentrations, (remaining_space + 1) // 2)
        _extend_exactly(result, pps, remaining_space // 2)
        assert len(result) == max_size

    return np.asarray(result[:max_size])


def _extend_exactly(lhs, rhs, amount: int, padding: int = 0):
    """Extend lhs, with exactly amount elements from rhs.  If there are
    not enough elements, lhs is padded to the correct amount with padding.
    """
    padding_len = amount - len(rhs)  # May be negative
    lhs.extend(rhs[:amount])
    lhs.extend([padding] * padding_len)  # NO-OP if padding_len is negative


DEFAULT_TIMING_FEATURES = [
    # Interarrival stats
    'interarrival::in::max', 'interarrival::out::max',
    'interarrival::overall::max', 'interarrival::in::mean',
    'interarrival::out::mean', 'interarrival::overall::mean',
    'interarrival::in::std', 'interarrival::out::std',
    'interarrival::overall::std', 'interarrival::in::percentile-75',
    'interarrival::out::percentile-75', 'interarrival::overall::percentile-75',
    # Timestamp percentiles
    'time::in::percentile-25', 'time::in::percentile-50',
    'time::in::percentile-75', 'time::in::percentile-100',
    'time::out::percentile-25', 'time::out::percentile-50',
    'time::out::percentile-75', 'time::out::percentile-100',
    'time::overall::percentile-25', 'time::overall::percentile-50',
    'time::overall::percentile-75', 'time::overall::percentile-100',
    # Packet counts
    'packet-counts::in', 'packet-counts::out', 'packet-counts::overall',
    # First and last 30 packet concentrations
    'first-30::packet-counts::in', 'first-30::packet-counts::out',
    'last-30::packet-counts::in', 'last-30::packet-counts::out',
    # Some concentration stats
    'concentration-stats::std::out', 'concentration-stats::mean::out',
    # Some packets per-second stats
    'pps::mean', 'pps::std',
    # Packet ordering statistics
    'packet-order::out::mean', 'packet-order::in::mean',
    'packet-order::out::std', 'packet-order::in::std',
    # Concentration stats ctd.
    'concentration-stats::median::out',
    # Remaining packet per second stats
    'pps::median', 'pps::min', 'pps::max',
    # Concentration stats ctd.
    'concentration-stats::max::out',
    # Fraction of packets in each direction
    'fraction-incoming', 'fraction-outgoing',
]

DEFAULT_SIZE_FEATURES = [
    # Total sizes
    'total-size::in', 'total-size::out', 'total-size::overall',
    # Size statistics
    'size-stats::in::mean', 'size-stats::in::max', 'size-stats::in::var',
    'size-stats::in::std',
    'size-stats::out::mean', 'size-stats::out::max', 'size-stats::out::var',
    'size-stats::out::std',
    'size-stats::overall::mean', 'size-stats::overall::max',
    'size-stats::overall::var', 'size-stats::overall::std',
]

ALL_DEFAULT_FEATURES = list(itertools.chain(
    DEFAULT_TIMING_FEATURES,
    [f'alt-conc::{i}' for i in range(20)],
    [f'alt-pps::{i}' for i in range(20)],
    ['alt-pps::sum'],
    DEFAULT_SIZE_FEATURES,
    [f'conc::{i}' for i in range((DEFAULT_NUM_FEATURES - 102 + 1) // 2)],
    [f'pps::{i}' for i in range((DEFAULT_NUM_FEATURES - 102) // 2)]
))
