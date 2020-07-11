"""Extract features from traces for website fingerprinting
classification.
"""
from typing import Sequence
import numpy as np


def vectorize_trace(trace: Sequence, n_features: int):
    """Vectorize the trace, extracting features used by the p1-FP(C)
    classifier.

    Traces longer than n_features are truncated, traces shorter are
    padded with zero.
    """
    # Truncate the trace to the required length if longer
    trace = trace[:n_features]

    # Convert and padd
    padding_len = n_features - len(trace)
    return np.array([packet[1] * packet[2] for packet in trace]
                    # A negative padding_len results in no padding being added
                    + [0] * padding_len)


def vectorize_traces(traces: Sequence[Sequence], n_features: int) -> np.ndarray:
    """Vectorize multiple traces, extracting features used by the p1-FP(C)
    classifier.

    Traces longer than n_features are truncated, traces shorter are
    padded with zero.
    """
    return np.array([vectorize_trace(trace, n_features) for trace in traces])
