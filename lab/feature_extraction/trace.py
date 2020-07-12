"""Extract features from traces for website fingerprinting
classification.
"""
import enum
import numpy as np


def extract_sizes(X) -> np.ndarray:
    """Extract the packet sizes from a sequence of padded traces.

    Parameters:
    -----------
    X : array-like with shape (n_samples, n_packets, 2)
        A non-ragged sequence of traces. Use pad_traces first if you
        have a ragged se
    """
    X = check_traces(X)
    # Make a copy so that we do not keep the underlying traces alive
    return X[:, :, 1].copy()


def pad_traces(X):
    """Pad a sequence of traces to the length of the longest trace.
    """
    max_len = max(len(trace) for trace in X)
    # Use the same dtype as the rows
    sample_trace = np.asarray(X[0])
    n_fields = sample_trace.shape[1]

    result = np.zeros((len(X), max_len, n_fields), dtype=sample_trace.dtype)
    for i, trace in enumerate(X):
        result[i, :len(trace)] = trace
    return result


def check_traces(X) -> np.ndarray:
    """Check that the traces have the correct shape and return a numpy
    array.
    """
    X = np.asarray(X)
    if len(X.shape) != 3:
        raise ValueError(
            "A trace sequence should be a 3-dimension array-like. Actual shape "
            f"is {X.shape}. Ensure it's not jagged.")
    return X


def extract_interarrival_times(X) -> np.ndarray:
    """Extract the interarrival times from a sequence of padded traces.

    Parameters:
    -----------
    X : array-like with shape (n_samples, n_packets, 2)
        A non-ragged sequence of traces. Use pad_traces first if you
        have a ragged se
    """
    X = check_traces(X)

    # Make a copy so that we do not keep the underlying traces alive
    times = X[:, :, 0].copy()
    # Compute the interarrival time
    times[:, 1:] = times[:, 1:] - times[:, :-1]
    # Ensure that the matrix is positive. The computation may cause the last
    # packet time to be subtracted from a padding packet resulting in a negative
    # interarrival time.
    times[times < 0] = 0

    return times


class Metadata(enum.Flag):
    """Supported trace metadata.
    """
    UNSPECIFIED = 0

    PACKET_COUNT = enum.auto()
    OUTGOING_COUNT = enum.auto()
    INCOMING_COUNT = enum.auto()
    INCOMING_RATIO = enum.auto()
    OUTGOING_RATIO = enum.auto()

    DURATION = enum.auto()
    DURATION_PER_PACKET = enum.auto()

    # These have to be placed last, or enum.auto() reassigns values
    COUNT_METADATA = PACKET_COUNT | OUTGOING_COUNT | INCOMING_COUNT \
        | OUTGOING_RATIO | INCOMING_RATIO
    TIME_METADATA = DURATION | DURATION_PER_PACKET


def extract_metadata(
    X, metadata: Metadata = Metadata.UNSPECIFIED
) -> np.ndarray:
    """Extract metadata from the traces.  If unspecified, all metadata
    will be returned.
    """
    X = check_traces(X)
    # Create views for times and sizes
    times = X[:, :, 0]
    sizes = X[:, :, 1]

    # Include all features if unspecified
    metadata = metadata or ~Metadata.UNSPECIFIED

    results = {}

    results[Metadata.PACKET_COUNT] = np.sum((sizes != 0), axis=1)
    results[Metadata.OUTGOING_COUNT] = np.sum((sizes > 0), axis=1)
    results[Metadata.INCOMING_COUNT] = np.sum((sizes < 0), axis=1)
    results[Metadata.OUTGOING_RATIO] = \
        results[Metadata.OUTGOING_COUNT] / results[Metadata.PACKET_COUNT]
    results[Metadata.INCOMING_RATIO] = \
        results[Metadata.INCOMING_COUNT] / results[Metadata.PACKET_COUNT]

    results[Metadata.DURATION] = np.amax(times, axis=1)
    results[Metadata.DURATION_PER_PACKET] = \
        results[Metadata.DURATION] / results[Metadata.PACKET_COUNT]

    order = [
        Metadata.PACKET_COUNT, Metadata.OUTGOING_COUNT, Metadata.INCOMING_COUNT,
        Metadata.OUTGOING_RATIO, Metadata.INCOMING_RATIO,
        Metadata.DURATION, Metadata.DURATION_PER_PACKET,
    ]
    return np.transpose(list(results[m] for m in order if m in metadata))
