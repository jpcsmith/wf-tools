"""Tests for feature extraction."""
# pylint: disable=invalid-name
from typing import Tuple

import pytest
import numpy as np

from lab.feature_extraction.trace import (
    extract_sizes, extract_interarrival_times, pad_traces,
    extract_metadata, Metadata
)


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> Tuple[list, np.ndarray, np.ndarray]:
    """Returns a simple list of traces."""
    traces = [
        [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)],
        [(0, 1300), (0.015, 1350), (0.025, 1200), (0, 0)],
        [(0, 1200), (0.02, 1350), (0, 0), (0, 0)],
    ]
    expected_features = np.array([
        [1350, 1350, -600, 70],
        [1300, 1350, 1200, 0],
        [1200, 1350, 0, 0]
    ])
    expected_interarrivals = np.array([
        [0, 0.010, 0.01, 0.01],
        [0, 0.015, 0.01, 0.00],
        [0, 0.020, 0.00, 0.00]
    ])
    return traces, expected_features, expected_interarrivals


def test_extract_interarrival_times(sample_traces):
    """It should extract the interarrival times from the traces
    and pad to the specified dimension.
    """
    traces, _, expected_times = sample_traces

    np.testing.assert_allclose(
        extract_interarrival_times(traces), expected_times)


def test_extract_sizes(sample_traces):
    """It extract the sizes from the traces.
    """
    traces, features, *_ = sample_traces
    np.testing.assert_array_equal(extract_sizes(traces), features)


def test_pad_traces():
    """It should pad the traces to the length of the longest trace."""
    np.testing.assert_allclose(
        pad_traces([
            [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)],
            [(0, 1300), (0.015, 1350), (0.025, 1200)],
            [(0, 1200), (0.02, 1350)],
        ]),
        np.array([
            [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)],
            [(0, 1300), (0.015, 1350), (0.025, 1200), (0.0, 0)],
            [(0, 1200), (0.02, 1350), (0.0, 0), (0.0, 0)],
        ]))


def test_extract_metadata_duration(sample_traces):
    """It should extract duration metadata from the traces."""
    traces, *_ = sample_traces
    np.testing.assert_allclose(
        extract_metadata(traces, metadata=Metadata.DURATION),
        # [[0.03, 0.03/4], [0.025, 0.025/3], [0.02, 0.02/2]])
        [[0.03], [0.025], [0.02]])


def test_extract_metadata_duration_per_packet(sample_traces):
    """It should extract duration per packet from the traces."""
    traces, *_ = sample_traces
    np.testing.assert_allclose(
        extract_metadata(traces, metadata=Metadata.DURATION_PER_PACKET),
        [[0.03/4], [0.025/3], [0.02/2]])


def test_extract_metadata_packet_count(sample_traces):
    """It should extract total packet count metadata from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.PACKET_COUNT),
        [[4], [3], [2]])


def test_extract_metadata_outgoing_count(sample_traces):
    """It should extract outgoing packet count metadata from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.OUTGOING_COUNT),
        [[3], [3], [2]])


def test_extract_metadata_incoming_count(sample_traces):
    """It should extract incoming packet count metadata from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.INCOMING_COUNT),
        [[1], [0], [0]])


def test_extract_metadata_incoming_ratio(sample_traces):
    """It should extract incoming packet ratio from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.INCOMING_RATIO),
        [[1/4], [0/3], [0/2]])


def test_extract_metadata_outgoing_ratio(sample_traces):
    """It should extract outgoing packet ratio from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.OUTGOING_RATIO),
        [[3/4], [3/3], [2/2]])


def test_extract_metadata_count_metadata(sample_traces):
    """It should extract all count metadata from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.COUNT_METADATA),
        [[4, 3, 1, 3/4, 1/4], [3, 3, 0, 3/3, 0], [2, 2, 0, 2/2, 0]])


def test_extract_metadata_size_metadata(sample_traces):
    """It should extract all size metadata from the traces."""
    np.testing.assert_allclose(
        extract_metadata(sample_traces[0], metadata=Metadata.SIZE_METADATA),
        np.transpose([
            # Sizes
            [3370, 3850, 2550],
            # Outgoing and incoming sizes
            [2770, 3850, 2550], [600, 0, 0],
            # Outgoing and incoming size ratio
            [2770/3370, 1, 1], [600/3370, 0, 0],
        ]))


def test_extract_metadata_unspecified(sample_traces):
    """It should return all of the metadata if unspecified."""
    n_features = 12
    assert extract_metadata(sample_traces[0]).shape == (3, n_features)
