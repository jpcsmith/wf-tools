"""Code form the paper:

    S. Li, H. Guo, and N. Hopper, "Measuring Information Leakage in
    Website Fingerprinting Attacks and Defenses," in Proceedings of the
    2018 ACM SIGSAC Conference on Computer and Communications Security,
    Toronto, Canada, 2018, pp. 1977–1992, doi: 10.1145/3243734.3243832.

The original code can be found at:

    https://github.com/s0irrlor7m/InfoLeakWebsiteFingerprint.

The code was converted from python 2 to python 3 using the 2to3-2.7 tool,
and is covered under the CRAPL v0 BETA 1 license. See
http://matt.might.net/ for further information.
"""
from typing import Sequence, List
import numpy
from .Extract import extract

__all__ = ["extract_features", "FEATURE_NAMES"]


def extract_features(
    timestamps: Sequence[float], sizes: Sequence[float]
) -> numpy.ndarray:
    """Return feautres extracted from timestamps and sizes.  To change
    the set of features extracted, see li2018measuring.Param.

    Missing features are returned as NaN.
    """
    features: List[float] = []

    extract(timestamps, sizes, features)

    for i, element in enumerate(features):
        if element == "X":
            features[i] = numpy.NaN
    return numpy.asarray(features, dtype=float)


def _repeat_unique(name: str, lower: int, upper: int):
    return [f"{name}::{i}" for i in range(upper - lower + 1)]


FEATURE_NAMES = [
    # --- Packet Count (1-13)
    "Packet Count::total pkt count",
    "Packet Count::outgoing pkt count",
    "Packet Count::incoming pkt count",
    "Packet Count::outgoing count/total count",
    "Packet Count::incoming count/total count",
    "Packet Count::Rounding by WPES11::0",
    "Packet Count::Rounding by WPES11::1",
    "Packet Count::Rounding by WPES11::2",
    "Packet Count::Rounding by WPES11::3",
    "Packet Count::Rounding by WPES11::4",
    "Packet Count::Pkt Size Sum::0",
    "Packet Count::Pkt Size Sum::1",
    "Packet Count::Pkt Size Sum::2",

    # --- Time Statistics (14-37)
    *_repeat_unique("Time Statistics::Total: Inter-Pkt statistics", 14, 17),
    *_repeat_unique("Time Statistics::Outgoing: Inter-Pkt statistics", 18, 21),
    *_repeat_unique("Time Statistics::Incoming: Inter-Pkt statistics", 22, 25),
    *_repeat_unique("Time Statistics::Total: Trans. Statistics", 26, 29),
    *_repeat_unique("Time Statistics::Outgoing: Trans. Statistics", 30, 33),
    *_repeat_unique("Time Statistics::Incoming: Trans. Statistics", 34, 37),

    # --- Packet Ordering (NGRAM) (38-161)
    *_repeat_unique("Packet Ordering (NGRAM)::2-Gram", 38, 41),
    *_repeat_unique("Packet Ordering (NGRAM)::3-Gram", 42, 49),
    *_repeat_unique("Packet Ordering (NGRAM)::4-Gram", 50, 65),
    *_repeat_unique("Packet Ordering (NGRAM)::5-Gram", 66, 97),
    *_repeat_unique("Packet Ordering (NGRAM)::6-Gram", 98, 161),

    # --- Packet Ordering (TRANS_POSITION) (162-765)
    *_repeat_unique("Packet Ordering (TRANS_POSITION)::Outgoing", 162, 461),
    "Packet Ordering (TRANS_POSITION)::Outgoing::std",
    "Packet Ordering (TRANS_POSITION)::Outgoing::mean",
    *_repeat_unique("Packet Ordering (TRANS_POSITION)::Incoming", 464, 763),
    "Packet Ordering (TRANS_POSITION)::Incoming::std",
    "Packet Ordering (TRANS_POSITION)::Incoming::mean",

    # --- Intervals (KNN) (766-1365)
    *_repeat_unique("Intervals (KNN)::Incoming", 766, 1065),
    *_repeat_unique("Intervals (KNN)::Outgoing", 1066, 1365),

    # --- Intervals (ICICS) (1366-1967)
    *_repeat_unique("Intervals (ICICS)::Incoming", 1366, 1666),
    *_repeat_unique("Intervals (ICICS)::Outgoing", 1667, 1967),

    # --- Intervals (WPES11) (1968-2553)
    *_repeat_unique("Intervals (WPES11)::Incoming", 1968, 2260),
    *_repeat_unique("Intervals (WPES11)::Outgoing", 2261, 2553),

    # --- Packet Distribution (2554-2778)
    *_repeat_unique("Packet Distribution::Outgoing", 2554, 2753),
    "Packet Distribution::Outgoing::std",
    "Packet Distribution::Outgoing::mean",
    "Packet Distribution::Outgoing::median",
    "Packet Distribution::Outgoing::max",
    *_repeat_unique("Packet Distribution::alternative", 2758, 2777),
    "Packet Distribution::alternative::sum",

    # --- BURST (2779-2789)
    *_repeat_unique("BURST::ave + ave + num", 2779, 2781),
    *_repeat_unique("BURST::(>5) + (>10) + (>15)", 2782, 2784),
    *_repeat_unique("BURST::first 5 bursts", 2785, 2789),

    # --- Other (2790-3043)
    *_repeat_unique("FIRST20", 2790, 2809),
    *_repeat_unique("FIRST30_PKT_NUM", 2810, 2811),
    *_repeat_unique("LAST30_PKT_NUM", 2812, 2813),
    *_repeat_unique("PKT_PER_SECOND", 2814, 2939),
    *_repeat_unique("CUMUL", 2940, 3043)
]
