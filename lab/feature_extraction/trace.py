"""Extract features from traces for website fingerprinting
classification.
"""
import numpy as np


def extract_sizes(trace, dimension: int = 0) -> np.ndarray:
    """Extract the signed sizes from the trace.

    Parameters
    ----------
    trace : array-like of shape (n_packets, 2)
        trace is a sequence of n_packets, where each packet stores the
        timestamp and signed size.

    dimension :
        If dimension is greater than zero, truncate or pad the resulting
        feature vector the specified dimension.
    """
    assert dimension >= 0

    # Truncate the trace to the required length if longer
    if dimension > 0:
        trace = trace[:dimension]

    # Convert and pad
    padding_len = dimension - len(trace)
    # A negative padding_len results in no padding being added
    return np.array([pkt[1] for pkt in trace] + [0] * padding_len)


def extract_sizes_3d(X, dimension: int = 0) -> np.ndarray:
    """Convenience method around extract_sizes for a sequence of traces.

    Parameters:
    -----------
    X : Sequence of array-likes each with shape (n_packets(i), 2)
        A possibly ragged sequence of traces.

    dimension :
        If dimension is greater than zero, truncate or pad the resulting
        feature vector the specified dimension. If not specified, the
        dimension will be the length of the longest trace in X.
    """
    assert dimension >= 0
    dimension = dimension or max(len(trace) for trace in X)
    return np.array([extract_sizes(trace, dimension) for trace in X])


def extract_interarrival_times(trace, dimension: int = 0) -> np.ndarray:
    """Extract the interarrival times from the trace.

    Parameters
    ----------
    trace : array-like of shape (n_packets, 2)
        trace is a sequence of n_packets, where each packet stores the
        timestamp and signed size.

    dimension :
        If dimension is greater than zero, truncate or pad the resulting
        feature vector the specified dimension.
    """
    assert dimension >= 0

    # Truncate the trace to the required length if longer
    if dimension > 0:
        trace = trace[:dimension]

    # A negative padding_len results in no padding being added
    times = np.array([pkt[0] for pkt in trace])
    # The times should start from zero
    assert len(times) == 0 or times[0] == 0

    # Calculate the interarrival times
    times[1:] = times[1:] - times[:-1]

    # Pad the interarrival times if necessary. Truncated would have already
    # occured if necessary
    if dimension > 0 and len(times) != dimension:
        padded_times = np.zeros(dimension, dtype=times.dtype)
        padded_times[:len(times)] = times
        return padded_times
    return times


def extract_interarrival_times_3d(X, dimension: int = 0) -> np.ndarray:
    """Convenience method around extract_interarrival_times for a
    sequence of traces.

    Parameters:
    -----------
    X : Sequence of array-likes each with shape (n_packets(i), 2)
        A possibly ragged sequence of traces.

    dimension :
        If dimension is greater than zero, truncate or pad the resulting
        feature vector the specified dimension. If not specified, the
        dimension will be the length of the longest trace in X.
    """
    assert dimension >= 0
    dimension = dimension or max(len(trace) for trace in X)
    return np.array([extract_interarrival_times(trace, dimension)
                     for trace in X])
