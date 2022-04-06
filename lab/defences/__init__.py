"""Website Fingerprinting defence simulations."""
import logging
import itertools
from typing import List, Tuple, Union

import numpy as np

__all__ = ["PACKET_DTYPE"]

#: Data type used for the resulting traces
PACKET_DTYPE = np.dtype([("time", "<f8"), ("size", "<i8")])


def _sort_trace(trace) -> np.ndarray:
    """Sort a trace by time with with ties broken by outgoing packets
    first.
    """
    assert trace.dtype == PACKET_DTYPE, "invalid trace dtype"
    key_trace = np.array(
        list(map(lambda x: (x["time"], -x["size"]), trace)), dtype=PACKET_DTYPE
    )
    return trace[key_trace.argsort()]
