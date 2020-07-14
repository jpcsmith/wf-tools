"""Extract features from traces for website fingerprinting
classification.
"""
import enum
import numpy as np


DEFAULT_PACKET_DTYPE = np.dtype([("timestamp", "f8"), ("signed_size", "i4")])


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
    return X[:, :]["signed_size"].copy()


def pad_traces(X):
    """Pad a sequence of traces to the length of the longest trace.
    """
    max_len = max(len(trace) for trace in X)
    # Use the same dtype as the rows
    sample_trace = np.asarray(X[0])
    try:
        shape = (len(X), max_len, sample_trace.shape[1])
    except IndexError:
        shape = (len(X), max_len)  # type: ignore

    result = np.zeros(shape, dtype=sample_trace.dtype)
    for i, trace in enumerate(X):
        result[i, :len(trace)] = trace
    return result


def check_traces(X) -> np.ndarray:
    """Check that the traces have the correct shape and return a numpy
    array.
    """
    X = np.asarray(X, dtype=DEFAULT_PACKET_DTYPE)
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
    times = X[:, :]["timestamp"].copy()
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

    TRANSFER_SIZE = enum.auto()
    OUTGOING_SIZE = enum.auto()
    INCOMING_SIZE = enum.auto()
    OUTGOING_SIZE_RATIO = enum.auto()
    INCOMING_SIZE_RATIO = enum.auto()

    # These have to be placed last, or enum.auto() reassigns values
    COUNT_METADATA = PACKET_COUNT | OUTGOING_COUNT | INCOMING_COUNT \
        | OUTGOING_RATIO | INCOMING_RATIO
    TIME_METADATA = DURATION | DURATION_PER_PACKET
    SIZE_METADATA = TRANSFER_SIZE | OUTGOING_SIZE | INCOMING_SIZE \
        | OUTGOING_SIZE_RATIO | INCOMING_SIZE_RATIO


def extract_metadata(
    X, metadata: Metadata = Metadata.UNSPECIFIED
) -> np.ndarray:
    """Extract metadata from the traces.  If unspecified, all metadata
    will be returned.
    """
    X = check_traces(X)
    # Create views for times and sizes
    times = X[:, :]["timestamp"]
    sizes = X[:, :]["signed_size"]

    # Include all metadata if unspecified
    metadata = metadata or ~Metadata.UNSPECIFIED

    results = {}

    results[Metadata.PACKET_COUNT] = np.sum((sizes != 0), axis=1)
    if Metadata.COUNT_METADATA | metadata:
        results[Metadata.OUTGOING_COUNT] = np.sum((sizes > 0), axis=1)
        results[Metadata.INCOMING_COUNT] = np.sum((sizes < 0), axis=1)
        results[Metadata.OUTGOING_RATIO] = \
            results[Metadata.OUTGOING_COUNT] / results[Metadata.PACKET_COUNT]
        results[Metadata.INCOMING_RATIO] = \
            results[Metadata.INCOMING_COUNT] / results[Metadata.PACKET_COUNT]

    if Metadata.TIME_METADATA | metadata:
        results[Metadata.DURATION] = np.amax(times, axis=1)
        results[Metadata.DURATION_PER_PACKET] = \
            results[Metadata.DURATION] / results[Metadata.PACKET_COUNT]

    if Metadata.SIZE_METADATA | metadata:
        results[Metadata.TRANSFER_SIZE] = np.sum(np.abs(sizes), axis=1)
        results[Metadata.OUTGOING_SIZE] = np.sum(np.where(sizes > 0, sizes, 0),
                                                 axis=1)
        results[Metadata.INCOMING_SIZE] = np.sum(
            np.where(sizes < 0, np.abs(sizes), 0), axis=1)
        results[Metadata.OUTGOING_SIZE_RATIO] = \
            results[Metadata.OUTGOING_SIZE] / results[Metadata.TRANSFER_SIZE]
        results[Metadata.INCOMING_SIZE_RATIO] = \
            results[Metadata.INCOMING_SIZE] / results[Metadata.TRANSFER_SIZE]

    order = [
        Metadata.PACKET_COUNT, Metadata.OUTGOING_COUNT, Metadata.INCOMING_COUNT,
        Metadata.OUTGOING_RATIO, Metadata.INCOMING_RATIO,

        Metadata.DURATION, Metadata.DURATION_PER_PACKET,

        Metadata.TRANSFER_SIZE, Metadata.OUTGOING_SIZE, Metadata.INCOMING_SIZE,
        Metadata.OUTGOING_SIZE_RATIO, Metadata.INCOMING_SIZE_RATIO,
    ]
    return np.transpose(list(results[m] for m in order if m in metadata))
