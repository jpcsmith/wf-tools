"""Simulation of the Tamaraw defence for a given trace.

For more details, see the paper:

  Cai, X., Nithyanand, R., Wang, T., Johnson, R., & Goldberg, I. (2014).
  A Systematic Approach to Developing and Evaluating Website
  Fingerprinting Defenses. CCS 2014. https://doi.org/10.1145/2660267.2660362

"""
import itertools
from typing import List, Tuple

import numpy as np

from . import PACKET_DTYPE, _sort_trace


def simulate(
    trace: List[Tuple[float, int]],
    packet_size: int = 750,
    rate_in: float = 0.006,
    rate_out: float = 0.02,
    pad_multiple: int = 100,
):
    """Simulate the Tamaraw defence [1] for the given trace.

    The parameter packet_size is the size of each simulated packet.

    The parameters rate_in and rate_out are in seconds per packet and
    define the frequency of the packets.

    Based on the implementation of Cai et al. [2].

    References:

      [1] Cai et al. (2014). A Systematic Approach to Developing and
          Evaluating Website Fingerprinting Defenses. ACM SIGSAC CCS

      [2] http://home.cse.ust.hk/~taow/wf/defenses/tamaraw.py

    """
    packets = np.asarray(trace, dtype=PACKET_DTYPE)
    assert np.all(packets["size"] != 0), "packet sizes must not be zero"

    # Sort the packets by time in case the capture has out-of-order packets
    packets.sort()
    # Set all the timestamps relative to the first
    packets["time"] -= packets["time"][0]

    is_out = (packets["size"] > 0)
    result = np.append(
        _tamaraw_uni(packets[is_out], packet_size, rate_out, pad_multiple),
        _tamaraw_uni(packets[~is_out], -packet_size, rate_in, pad_multiple),
    )
    return _sort_trace(result)


def _tamaraw_uni(
    trace: np.ndarray,
    packet_size: int,
    rate: float,
    pad_multiple: int,
):
    """Simulate the Tamaraw defence for a single direction of packets.

    The trace must be sorted by timestamp and be relative to zero.
    """
    assert packet_size != 0, "packet_size must be non-negative"
    assert trace.dtype == PACKET_DTYPE, "invalid trace dtype"
    assert (trace["size"] > 0).all() or (trace["size"] < 0).all()

    result = []
    trace = np.copy(trace)

    trace_idx = 0
    time_idx = itertools.count(start=0)
    while trace_idx < len(trace):
        # Compute the timestamp
        timestamp = next(time_idx) * rate
        # We send a packet at every timestamp
        result.append((timestamp, packet_size))

        # Consume packets from the trace to make up the sent packet
        to_send = packet_size
        while (
            trace_idx < len(trace)
            and trace[trace_idx]["time"] <= timestamp
            and to_send != 0
        ):
            sent = (to_send if abs(to_send) < abs(trace[trace_idx]["size"])
                    else trace[trace_idx]["size"])
            trace[trace_idx]["size"] -= sent
            to_send -= sent

            # If we fully consumed a packet, we can move to the next
            if trace[trace_idx]["size"] == 0:
                trace_idx += 1

    # Add the padding packets
    while len(result) % pad_multiple != 0:
        timestamp = next(time_idx) * rate
        result.append((timestamp, packet_size))

    return np.asarray(result, dtype=PACKET_DTYPE)
