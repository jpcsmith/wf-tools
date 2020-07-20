"""Extract features from traces for website fingerprinting
classification.
"""
import enum
from typing import Sequence, Union
import numpy as np


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


def ensure_non_ragged(
    X: Sequence, dimension: int = 0, copy: bool = False
) -> np.ndarray:
    """Pad a sequence to the length of the longest trace feature.

    If dimension is greater than zero, the array resulting ndarray will
    have shape[1] == dimension, i.e. truncated if too long and padded if
    too short.

    Will avoid copying if copy=True and dimension is 0.
    """
    # Check if it's already non-jagged
    result = np.array(X, copy=copy)
    if len(result.shape) == 2 and dimension == 0:
        return result

    dimension = dimension or max(len(row) for row in X)

    # Use the same dtype as the rows
    sample_row = np.asarray(X[0])

    if len(sample_row.shape) != 1:
        raise ValueError("Input must be at most 2D.")

    result = np.zeros((len(X), dimension), dtype=sample_row.dtype)
    for i, trace in enumerate(X):
        # Avoid slicing the list if not necessary
        if len(trace) <= dimension:
            result[i, :len(trace)] = trace
        else:
            result[i, :] = trace[:dimension]
    return result


def extract_interarrival_times(X, dimension: int = 0) -> np.ndarray:
    """Extract the interarrival times from a sequence of potentially
    ragged timestamps.

    Parameters:
    -----------
    X : array-like with shape (n_samples, n_timestamps)
        A potentially ragged sequence of timestamps.
    """
    # Make a copy so that we do not modify the original
    times = ensure_non_ragged(X, dimension=dimension, copy=True)

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
    # Metadata such as total, incoming, and outgoing packet counts and their
    # ratios
    COUNT_METADATA = enum.auto()
    # Metadata such as duration and duration per packet
    TIME_METADATA = enum.auto()
    # Metadta such as total, incoming, outgoing and packet size totals and their
    # ratios
    SIZE_METADATA = enum.auto()


def extract_metadata(
    sizes: Union[Sequence[Sequence], np.ndarray, None] = None,
    timestamps: Union[Sequence[Sequence], np.ndarray, None] = None,
    metadata: Metadata = Metadata.UNSPECIFIED
) -> np.ndarray:
    """Extract metadata from the traces.  If unspecified, all metadata
    will be returned.

    Requires sizes or timestamps depending on the metadata requested.
    """
    # Unspecified is zero, in which case we set to all
    metadata = metadata or ~Metadata.UNSPECIFIED

    if (Metadata.TIME_METADATA in metadata) and timestamps is None:
        raise ValueError("Time features are required for time metadata.")
    if (Metadata.COUNT_METADATA in metadata) and sizes is None:
        raise ValueError("Size features are required for packet counts.")
    if (Metadata.SIZE_METADATA in metadata) and sizes is None:
        raise ValueError("Size features are required for size metadata.")

    if sizes is not None:
        sizes = ensure_non_ragged(sizes)
    if timestamps is not None:
        timestamps = ensure_non_ragged(timestamps)

    results = {}

    if Metadata.COUNT_METADATA in metadata:
        results["packet_count"] = np.sum((sizes != 0), axis=1)
        results["outgoing_count"] = np.sum((sizes > 0), axis=1)
        results["incoming_count"] = np.sum((sizes < 0), axis=1)
        results["outgoing_ratio"] = (
            results["outgoing_count"] / results["packet_count"])
        results["incoming_ratio"] = (
            results["incoming_count"] / results["packet_count"])

    if Metadata.SIZE_METADATA in metadata:
        results["transfer_size"] = np.sum(np.abs(sizes), axis=1)
        results["outgoing_size"] = np.sum(
            np.where(sizes > 0, sizes, 0), axis=1)  # type: ignore
        results["incoming_size"] = np.sum(
            np.where(sizes < 0, np.abs(sizes), 0), axis=1)  # type: ignore
        results["outgoing_size_ratio"] = \
            results["outgoing_size"] / results["transfer_size"]
        results["incoming_size_ratio"] = \
            results["incoming_size"] / results["transfer_size"]

    if Metadata.TIME_METADATA in metadata:
        # Count the number of non-zero times plus the initial zero packet
        packet_count = np.sum((timestamps > 0), axis=1) + 1
        results["duration"] = np.amax(timestamps, axis=1)
        results["duration_per_packet"] = results["duration"] / packet_count

    return np.transpose(list(results.values()))
