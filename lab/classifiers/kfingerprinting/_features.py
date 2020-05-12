"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

Minor adjustments to the work of the original authors from the paper. The
original can be found at https://github.com/jhayes14/k-FP.
"""
import math
import itertools
from typing import Tuple, Union, Sequence

import numpy as np

from lab.trace import Direction, Trace

DEFAULT_NUM_FEATURES = 165


# --------------------
# Non-feeder functions
# --------------------
def split_in_out(list_data: Trace, check: bool = True) -> Tuple[Trace, Trace]:
    """Returns a tuple of the packets in the (incoming, outgoing) subtraces.

    Raise AssertionError if check is true and the trace has no incoming or no
    outgoing packets.
    """
    incoming = [pkt for pkt in list_data if pkt.direction == Direction.IN]
    outgoing = [pkt for pkt in list_data if pkt.direction == Direction.OUT]
    if check:
        assert incoming and outgoing
    return (incoming, outgoing)


# -------------
# TIME FEATURES
# -------------
def _inter_pkt_time(list_data):
    if len(list_data) == 1:
        return [0.0, ]
    times = [x[0] for x in list_data]
    temp = []
    for elem, next_elem in zip(times, times[1:]+[times[0]]):
        temp.append(next_elem-elem)
    return temp[:-1]


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
        'mean': np.mean(times) if times else 0,
        'max': max(times, default=0),
        'std': np.std(times) if times else 0,
        'percentile-75': np.percentile(times, 75) if times else 0
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
        times = [pkt[0] for pkt in trace]
        return {f'percentile-{p}': (np.percentile(times, p) if times else 0)
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
    packets_per_sec, _ = np.histogram([pkt.timestamp for pkt in overall],
                                      bins=n_seconds, range=(0, n_seconds))
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

def total_packet_sizes(overall: Trace) -> dict:
    """Return the total incoming, outgoing and overall packet sizes."""
    incoming, outgoing = split_in_out(overall)

    # Use absolute value in case the input sizes are signed
    result = {
        'total-size::in': sum(abs(pkt.size) for pkt in incoming),
        'total-size::out': sum(abs(pkt.size) for pkt in outgoing),
    }
    result['total-size::overall'] = result['total-size::in'] \
        + result['total-size::out']
    return result


def _packet_size_stats(trace: Trace) -> dict:
    sizes = [pkt.size for pkt in trace]
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
def extract_features(trace: Trace, max_size: int = DEFAULT_NUM_FEATURES) \
        -> Tuple[float, ...]:
    """Return a tuple of features of the specified size, according to the paper

        Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
        scalable website fingerprinting technique." 25th {USENIX} Security
        Symposium ({USENIX} Security 16). 2016.

    """
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

    return tuple(result[:max_size])


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
