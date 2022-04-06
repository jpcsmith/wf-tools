"""Code related to the website fingerprinting FRONT defence.

For more details see the paper

  Gong, J., & Wang, T. (2020). Zero-delay Lightweight Defenses against
  Website Fingerprinting. USENIX Security 20.
  https://www.usenix.org/conference/usenixsecurity20/presentation/gong

"""
import logging
from typing import Union

import numpy as np
import pandas as pd

from .. import tracev2 as trace
from . import PACKET_DTYPE, _sort_trace

_LOGGER = logging.getLogger(__name__)


def simulate(traffic: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Simulate the the traffic defended FRONT with the specified
    schedule.
    """
    assert traffic.dtype == PACKET_DTYPE
    assert schedule.dtype == PACKET_DTYPE

    data = (pd.DataFrame(np.concatenate((traffic, schedule)))
            .assign(is_outgoing=lambda df: df["size"] > 0)
            .groupby(["time", "is_outgoing"])
            .sum()
            .reset_index()
            .loc[:, ["time", "size"]])

    simulated = np.rec.fromarrays(
        [data["time"], data["size"]], dtype=PACKET_DTYPE
    )
    return trace.sort(simulated)


def generate_padding(
    max_client_packets: int = 2500,
    max_server_packets: int = 2500,
    packet_size: int = 1200,
    peak_minimum: float = 1.,
    peak_maximum: float = 14.,
    random_state: Union[np.random.Generator, int, None] = None,
):
    """Return padding packets sampled according to the FRONT defence
    of Gong & Wang [1].

    References:

      [1] Gong & Wang (2020). Zero-delay Lightweight Defenses against
          Website Fingerprinting. USENIX Security

    """
    rand = random_state
    if rand is None or isinstance(rand, int):
        rand = np.random.default_rng(rand)

    _LOGGER.debug("Sampling outgoing timestamps.")
    out_times = _sample_front_timestamps(
        max_client_packets, peak_minimum, peak_maximum, rand)
    _LOGGER.debug("Sampling incoming timestamps.")
    in_times = _sample_front_timestamps(
        max_server_packets, peak_minimum, peak_maximum, rand)

    trace = np.zeros(len(out_times) + len(in_times), dtype=PACKET_DTYPE)
    trace["time"][:len(out_times)] = out_times
    trace["time"][-len(in_times):] = in_times
    trace["size"][:len(out_times)] = packet_size
    trace["size"][-len(in_times):] = -packet_size

    return _sort_trace(trace)


def _sample_front_timestamps(max_packets, peak_minimum, peak_maximum, rand):
    n_packets = rand.integers(1, max_packets, dtype=int, endpoint=True)
    _LOGGER.debug("Sampled n_packets of %d from the interval [1, %d].",
                  n_packets, max_packets)
    peak = (peak_maximum - peak_minimum) * rand.random() + peak_minimum
    _LOGGER.debug("Sampled a rayleigh scale of %.3f from the interval"
                  " [%.3f, %.3f).", peak, peak_minimum, peak_maximum)
    return rand.rayleigh(peak, size=n_packets)
