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

__all__ = ["extract_features"]


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
