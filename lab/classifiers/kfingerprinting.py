"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

"""
import itertools
from typing import (
    List,
    Tuple,
    Iterable,
    Iterator,
    NamedTuple,
)

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


EncPacket = NamedTuple(
    'EncPacket', [('timestamp', float), ('size', int), ('incoming', bool)])
InterarrivalStats = NamedTuple(
    'InterarrivalStats',
    [('max', float), ('mean', float), ('std', float), ('upperq', float)])
# TODO: Preconding: Timestamps start from 0
Trace = List[EncPacket]


def interarrival_times(packets: Iterator[EncPacket]) -> Iterator[float]:
    """Returns the interarrival times between the provided packets."""
    # Get the first packet
    previous = next(packets, None)

    if previous is not None:
        for pkt in packets:
            yield pkt.timestamp - previous.timestamp
            previous = pkt


def _interarrival_stats(packets: Iterable[EncPacket]) -> InterarrivalStats:
    times = list(interarrival_times(iter(packets)))
    if not times:
        return InterarrivalStats(0, 0, 0, 0)
    return InterarrivalStats(max(times), np.mean(times), np.std(times),
                             np.percentile(times, 75))


def interarrival_stats(trace: Trace) -> Iterator[float]:
    """Extracts features for the max, mean, standard deviation and
    75th-percentile for the incoming, outgoing and entire trace.
    """
    return itertools.chain(
        _interarrival_stats(pkt for pkt in trace if pkt.incoming),
        _interarrival_stats(pkt for pkt in trace if not pkt.incoming),
        _interarrival_stats(trace),
    )


def _timestamp_percentiles(packets: Iterable[EncPacket]) \
        -> Tuple[float, float, float, float]:
    """Returns the 25th, 50th, 75th and 100th percentiles of the timestamps."""
    times = list(pkt.timestamp for pkt in packets)
    if not times:
        return (0, 0, 0, 0)
    return (np.percentile(times, 25), np.percentile(times, 50),
            np.percentile(times, 75), np.percentile(times, 100))


def timestamp_percentiles(trace: Trace) -> Iterator[float]:
    """Extracts features for 25th to 100th quartiles of timestamps for the
    incoming, outgoing and entire subtraces. Missing subtraces result in values
    of 0.
    """
    return itertools.chain(
        _timestamp_percentiles(pkt for pkt in trace if pkt.incoming),
        _timestamp_percentiles(pkt for pkt in trace if not pkt.incoming),
        _timestamp_percentiles(trace),
    )


def packet_counts(trace: Trace) -> Tuple[int, int, int]:
    """Returns the number of incoming, outgoing and total number of packets."""
    return (sum(pkt.incoming for pkt in trace),
            sum(not pkt.incoming for pkt in trace),
            len(trace))


def head_tail_concentration(trace: Trace, length: int = 30) \
        -> Tuple[int, int, int, int]:
    """Returns the number of incoming and outgoing packets in the first and
    last `length` packets of the trace. The result has the form
    (# inc. in head, # out. in head, # inc. in tail, # out. in tail)
    """
    return (sum(pkt.incoming for pkt in trace[:length]),
            sum(not pkt.incoming for pkt in trace[:length]),
            sum(pkt.incoming for pkt in trace[-length:]),
            sum(not pkt.incoming for pkt in trace[-length:]))


def _bin_trace(trace: pd.DataFrame, bin_size: int) -> DataFrameGroupBy:
    trace['group'] = np.arange(len(trace)) // bin_size
    return trace.groupby('group')


def outgoing_concentration_stats(trace: pd.DataFrame, bin_size: int = 20):
    """Compute statistics over the number of outgoing packets in each 20 packet
    interval.
    """
    frame = trace.copy()
    frame['outgoing'] = 1 - frame['incoming']
    return _bin_trace(frame, bin_size)['outgoing'].sum().describe()


def extract_features(trace: Trace) -> np.ndarray:
    """Extracts and returns the features as defined in the k-fingerprinting
    paper.
    """
    features = [
        interarrival_stats(trace),
        timestamp_percentiles(trace),
        iter(packet_counts(trace)),
        iter(head_tail_concentration(trace, length=30)),
    ]
    return list(itertools.chain.from_iterable(features))
