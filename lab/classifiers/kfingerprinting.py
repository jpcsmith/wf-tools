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
from pandas.api.types import CategoricalDtype


Directions = CategoricalDtype(['in', 'out', 'both'])


def interarrival_times(trace: pd.DataFrame) -> pd.DataFrame:
    """Returns the interarrival times between the provided packets."""
    return trace['timestamp'].diff().dropna()


def interarrival_stats(trace: pd.DataFrame) -> pd.DataFrame:
    """Extracts features for the max, mean, standard deviation and
    75th-percentile for the incoming, outgoing and entire trace.
    """
    assert not trace.empty

    times = trace.groupby('direction').apply(interarrival_times)
    stats = times.groupby('direction').describe()

    if 'in' not in stats.index:
        stats = stats.append(pd.Series([0, 0, 0], name='in').describe())
    if 'out' not in stats.index:
        stats = stats.append(pd.Series([0, 0, 0], name='out').describe())

    return stats.append(interarrival_times(trace).describe().rename('both'))


def timestamp_percentiles(trace: pd.DataFrame) -> pd.DataFrame:
    """Extracts features for 25th to 100th quartiles of timestamps for the
    incoming, outgoing and entire subtraces. Missing subtraces result in values
    of 0.
    """
    def _quantiles(frame):
        return frame.quantile([0.25, 0.5, 0.75, 1]).rename('quantile')

    assert not trace.empty

    percentiles = trace.groupby(
        'direction', observed=True)['timestamp'].apply(_quantiles).unstack()
    percentiles.index = percentiles.index.astype(Directions)

    if 'in' not in percentiles.index:
        percentiles.loc['in'] = _quantiles(pd.Series([0, 0, 0, 0]))
    if 'out' not in percentiles.index:
        percentiles.loc['out'] = _quantiles(pd.Series([0, 0, 0, 0]))
    return percentiles.append(_quantiles(trace['timestamp']).rename('both'))


def packet_counts(trace: pd.DataFrame) -> pd.Series:
    """Returns the number of incoming, outgoing and total number of packets."""
    counts = trace.groupby('direction', observed=False)['timestamp'].count()
    counts['both'] = len(trace)
    return counts

#
#
# def head_tail_concentration(trace: Trace, length: int = 30) \
#         -> Tuple[int, int, int, int]:
#     """Returns the number of incoming and outgoing packets in the first and
#     last `length` packets of the trace. The result has the form
#     (# inc. in head, # out. in head, # inc. in tail, # out. in tail)
#     """
#     return (sum(pkt.incoming for pkt in trace[:length]),
#             sum(not pkt.incoming for pkt in trace[:length]),
#             sum(pkt.incoming for pkt in trace[-length:]),
#             sum(not pkt.incoming for pkt in trace[-length:]))
#
#
# def _bin_trace(trace: pd.DataFrame, bin_size: int) -> DataFrameGroupBy:
#     trace['group'] = np.arange(len(trace)) // bin_size
#     return trace.groupby('group')
#
#
# def outgoing_concentration_stats(trace: pd.DataFrame, bin_size: int = 20):
#     """Compute statistics over the number of outgoing packets in each 20 packet
#     interval.
#     """
#     frame = trace.copy()
#     frame['outgoing'] = 1 - frame['incoming']
#     return _bin_trace(frame, bin_size)['outgoing'].sum().describe()
