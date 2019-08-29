"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

Minor adjustments to the work of the original authors from the paper. The
original can be found at https://github.com/jhayes14/k-FP.
"""
import math
from typing import (
    Tuple,
    NamedTuple,
    List,
    Union,
)
from typing_extensions import Literal, Final

import numpy as np

from lab.trace import (
    Packet,
    Direction,
    Trace
)


# ----------------
# Feeder functions
# ----------------
def neighborhood(iterable):
    iterator = iter(iterable)
    prev = (0)
    item = next(iterator)  # throws StopIteration if empty.
    for next_item in iterator:
        yield (prev, item, next_item)
        prev = item
        item = next_item
    yield (prev, item, None)


def chunkIt(seq, num):
    avg = len(seq) / num
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


# --------------------
# Non-feeder functions
# --------------------
def In_Out(list_data):
    In = []
    Out = []
    for p in list_data:
        if p[1] == -1:
            In.append(p)
        if p[1] == 1:
            Out.append(p)
    return In, Out


# -------------
# TIME FEATURES
# -------------
def inter_pkt_time(list_data):
    times = [x[0] for x in list_data]
    temp = []
    for elem, next_elem in zip(times, times[1:]+[times[0]]):
        temp.append(next_elem-elem)
    return temp[:-1]


def interarrival_times(list_data):
    In, Out = In_Out(list_data)
    IN = inter_pkt_time(In)
    OUT = inter_pkt_time(Out)
    TOTAL = inter_pkt_time(list_data)
    return IN, OUT, TOTAL


def interarrival_maxminmeansd_stats(list_data):
    interstats = []
    In, Out, Total = interarrival_times(list_data)
    if In and Out:
        avg_in = sum(In)/len(In)
        avg_out = sum(Out)/len(Out)
        avg_total = sum(Total)/len(Total)
        interstats.append((max(In), max(Out), max(Total), avg_in, avg_out,
                           avg_total, np.std(In), np.std(Out), np.std(Total),
                           np.percentile(In, 75), np.percentile(Out, 75),
                           np.percentile(Total, 75)))
    elif Out and not In:
        avg_out = sum(Out)/len(Out)
        avg_total = sum(Total)/len(Total)
        interstats.append((0, max(Out), max(Total), 0, avg_out, avg_total, 0,
                           np.std(Out), np.std(Total), 0,
                           np.percentile(Out, 75), np.percentile(Total, 75)))
    elif In and not Out:
        avg_in = sum(In)/len(In)
        avg_total = sum(Total)/len(Total)
        interstats.append((max(In), 0, max(Total), avg_in, 0, avg_total,
                           np.std(In), 0, np.std(Total), np.percentile(In, 75),
                           0, np.percentile(Total, 75)))
    else:
        interstats.extend(([0]*15))
    return interstats


def time_percentile_stats(Total: Trace):
    In, Out = In_Out(Total)
    In1 = [x[0] for x in In]
    Out1 = [x[0] for x in Out]
    Total1 = [x[0] for x in Total]
    STATS = []
    if In1:
        STATS.append(np.percentile(In1, 25))  # return 25th percentile
        STATS.append(np.percentile(In1, 50))
        STATS.append(np.percentile(In1, 75))
        STATS.append(np.percentile(In1, 100))
    if not In1:
        STATS.extend(([0]*4))
    if Out1:
        STATS.append(np.percentile(Out1, 25))  # return 25th percentile
        STATS.append(np.percentile(Out1, 50))
        STATS.append(np.percentile(Out1, 75))
        STATS.append(np.percentile(Out1, 100))
    if not Out1:
        STATS.extend(([0]*4))
    if Total1:
        STATS.append(np.percentile(Total1, 25))  # return 25th percentile
        STATS.append(np.percentile(Total1, 50))
        STATS.append(np.percentile(Total1, 75))
        STATS.append(np.percentile(Total1, 100))
    if not Total1:
        STATS.extend(([0]*4))
    return STATS


def number_pkt_stats(Total: Trace):
    In, Out = In_Out(Total)
    return len(In), len(Out), len(Total)


def first_and_last_30_pkts_stats(Total: Trace):
    first30 = Total[:30]
    last30 = Total[-30:]
    first30in = []
    first30out = []
    for p in first30:
        if p[1] == -1:
            first30in.append(p)
        if p[1] == 1:
            first30out.append(p)
    last30in = []
    last30out = []
    for p in last30:
        if p[1] == -1:
            last30in.append(p)
        if p[1] == 1:
            last30out.append(p)
    stats = []
    stats.append(len(first30in))
    stats.append(len(first30out))
    stats.append(len(last30in))
    stats.append(len(last30out))
    return stats


def pkt_concentration_stats(Total: Trace):
    """Concentration of outgoing packets in chunks of 20 packets."""
    chunks = [Total[x:x+20] for x in range(0, len(Total), 20)]
    concentrations = []
    for item in chunks:
        c = 0
        for p in item:
            if p[1] == 1:
                c += 1
        concentrations.append(c)
    return (np.std(concentrations),
            sum(concentrations)/len(concentrations),
            np.percentile(concentrations, 50), min(concentrations),
            max(concentrations), concentrations)


def number_per_sec(Total: Trace):
    """Average number packets sent and received per second."""
    last_time = Total[-1][0]
    last_second = math.ceil(last_time)
    temp = []
    l = []
    for i in range(1, int(last_second)+1):
        c = 0
        for p in Total:
            if p[0] <= i:
                c += 1
        temp.append(c)
    for prev, item, _ in neighborhood(temp):
        x = item - prev
        l.append(x)
    avg_number_per_sec = sum(l)/len(l)
    return (avg_number_per_sec, np.std(l), np.percentile(l, 50), min(l),
            max(l), l)


def avg_pkt_ordering_stats(Total: Trace):
    """Variant of packet ordering features
    from http://cacr.uwaterloo.ca/techreports/2014/cacr2014-05.pdf
    """
    c1 = 0
    c2 = 0
    temp1 = []
    temp2 = []
    for p in Total:
        if p[1] == 1:
            temp1.append(c1)
        c1 += 1
        if p[1] == -1:
            temp2.append(c2)
        c2 += 1
    avg_in = sum(temp1)/len(temp1)
    avg_out = sum(temp2)/len(temp2)

    return avg_in, avg_out, np.std(temp1), np.std(temp2)


def perc_inc_out(Total: Trace):
    In, Out = In_Out(Total)
    percentage_in = len(In)/len(Total)
    percentage_out = len(Out)/len(Total)
    return percentage_in, percentage_out


# ----------------
# FEATURE FUNCTION
# ----------------
# If size information available add them in to function below
def extract_features(trace: Trace, max_size: int = 175) -> Tuple[float, ...]:
    assert trace[0].timestamp == 0
    all_features = []

    # ------TIME--------
    intertimestats = [x for x in interarrival_maxminmeansd_stats(trace)[0]]
    timestats = time_percentile_stats(trace)
    number_pkts = list(number_pkt_stats(trace))
    thirtypkts = first_and_last_30_pkts_stats(trace)
    stdconc, avgconc, medconc, minconc, maxconc, conc = \
        pkt_concentration_stats(trace)

    avg_per_sec, std_per_sec, med_per_sec, min_per_sec, max_per_sec, per_sec = \
        number_per_sec(trace)
    avg_order_in, avg_order_out, std_order_in, std_order_out = \
        avg_pkt_ordering_stats(trace)
    perc_in, perc_out = perc_inc_out(trace)

    altconc = []
    alt_per_sec = []
    altconc = [sum(x) for x in chunkIt(conc, 70)]
    alt_per_sec = [sum(x) for x in chunkIt(per_sec, 20)]
    if len(altconc) == 70:
        altconc.append(0)
    if len(alt_per_sec) == 20:
        alt_per_sec.append(0)

    # TIME Features
    all_features.extend(intertimestats)
    all_features.extend(timestats)
    all_features.extend(number_pkts)
    all_features.extend(thirtypkts)
    all_features.append(stdconc)
    all_features.append(avgconc)
    all_features.append(avg_per_sec)
    all_features.append(std_per_sec)
    all_features.append(avg_order_in)
    all_features.append(avg_order_out)
    all_features.append(std_order_in)
    all_features.append(std_order_out)
    all_features.append(medconc)
    all_features.append(med_per_sec)
    all_features.append(min_per_sec)
    all_features.append(max_per_sec)
    all_features.append(maxconc)
    all_features.append(perc_in)
    all_features.append(perc_out)
    all_features.extend(altconc)
    all_features.extend(alt_per_sec)
    all_features.append(sum(altconc))
    all_features.append(sum(alt_per_sec))
    all_features.append(sum(intertimestats))
    all_features.append(sum(timestats))
    all_features.append(sum(number_pkts))

    # This is optional, since all other features are of equal size this gives
    # the first n features of this particular feature subset, some may be padded
    # with 0's if too short.
    all_features.extend(conc)
    all_features.extend(per_sec)

    while len(all_features) < max_size:
        all_features.append(0)
    features = all_features[:max_size]

    return tuple(features)