"""Tests for feature extraction."""
# pylint: disable=invalid-name
from typing import Tuple

import pytest
import numpy as np

from lab.feature_extraction.trace import (
    extract_sizes, extract_sizes_3d, extract_interarrival_times,
    extract_interarrival_times_3d
)


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> Tuple[list, np.ndarray, np.ndarray]:
    """Returns a simple list of traces."""
    traces = [
        [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)],
        [(0, 1300), (0.015, 1350), (0.025, 1200)],
        [(0, 1200), (0.02, 1350)],
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


def test_extract_interarrival_times_pad():
    """It should pad the extracted interarrival times to the specified
    dimension.
    """
    np.testing.assert_allclose(
        extract_interarrival_times(
            [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)], dimension=4),
        [0, 0.01, 0.01, 0.01])
    np.testing.assert_allclose(
        extract_interarrival_times([(0, 1300), (0.015, 1350), (0.025, 1200)],
                                   dimension=4),
        [0, 0.015, 0.01, 0.00])
    np.testing.assert_allclose(
        extract_interarrival_times(
            [(0, 1200), (0.02, 1350)], dimension=4), [0, 0.02, 0.00, 0.00])


def test_extract_interarrival_times():
    """It should extract the interarrival times from the traces."""
    np.testing.assert_allclose(
        extract_interarrival_times(
            [(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)]
        ), [0, 0.01, 0.01, 0.01])
    np.testing.assert_allclose(
        extract_interarrival_times([(0, 1300), (0.015, 1350), (0.025, 1200)]),
        [0, 0.015, 0.01])
    np.testing.assert_allclose(
        extract_interarrival_times([(0, 1200), (0.02, 1350)]), [0, 0.02])


def test_extract_interarrival_times_3d(sample_traces):
    """It should extract the interarrival times from the traces
    and pad to the specified dimension.
    """
    traces, _, expected_times = sample_traces

    np.testing.assert_allclose(
        extract_interarrival_times_3d(traces, dimension=4), expected_times)


def test_extract_interarrival_times_3d_truncate(sample_traces):
    """It should extract the interarrival times from the traces
    and truncate them to the specified dimension.
    """
    traces, _, expected_times = sample_traces

    np.testing.assert_allclose(
        extract_interarrival_times_3d(traces, dimension=2),
        expected_times[:, :2])


def test_extract_sizes_single_trace():
    """It should extract the sizes for a single trace."""
    np.testing.assert_array_equal(
        extract_sizes([(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)]),
        [1350, 1350, -600, 70])
    np.testing.assert_array_equal(
        extract_sizes([(0, 1300), (0.015, 1350), (0.025, 1200)]),
        [1300, 1350, 1200])
    np.testing.assert_array_equal(
        extract_sizes([(0, 1200), (0.02, 1350)]),
        [1200, 1350])


def test_extract_sizes_pad(sample_traces):
    """It should pad traces to the specified dimension."""
    traces, features, *_ = sample_traces
    features = np.concatenate(
        (features, np.zeros((features.shape[0], 2))), axis=1)

    for trace, expected in zip(traces, features):
        np.testing.assert_array_equal(
            extract_sizes(trace, dimension=6), expected)


def test_extract_sizes_truncate(sample_traces):
    """It should truncate traces to the specified dimension."""
    traces, features, *_ = sample_traces
    for trace, expected in zip(traces, features[:, :2]):
        np.testing.assert_array_equal(
            extract_sizes(trace, dimension=2), expected)


def test_extract_sizes_3d(sample_traces):
    """It should work with matices of traces as well.
    """
    traces, features, *_ = sample_traces
    np.testing.assert_array_equal(
        extract_sizes_3d(traces, dimension=3), features[:, :3])


def test_extract_sizes_3d_infer(sample_traces):
    """It should infer the dimension if not specified.
    """
    traces, features, *_ = sample_traces
    np.testing.assert_array_equal(extract_sizes_3d(traces), features)
