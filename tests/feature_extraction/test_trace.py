"""Tests for feature extraction."""
from typing import Tuple

import pytest
import numpy as np

from lab.feature_extraction.trace import extract_sizes, extract_sizes_3d


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> Tuple[list, np.ndarray]:
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
    return traces, expected_features


def test_extract_sizes_single_trace():
    """It should extract the sizes for a single trace."""
    assert np.array_equal(
        extract_sizes([(0, 1350), (0.01, 1350), (0.02, -600), (0.03, 70)]),
        [1350, 1350, -600, 70])
    assert np.array_equal(
        extract_sizes([(0, 1300), (0.015, 1350), (0.025, 1200)]),
        [1300, 1350, 1200])
    assert np.array_equal(
        extract_sizes([(0, 1200), (0.02, 1350)]),
        [1200, 1350])


def test_extract_sizes_pad(sample_traces):
    """It should pad traces to the specified dimension."""
    traces, features = sample_traces
    features = np.concatenate(
        (features, np.zeros((features.shape[0], 2))), axis=1)

    for trace, expected in zip(traces, features):
        assert np.array_equal(extract_sizes(trace, dimension=6), expected)


def test_extract_sizes_truncate(sample_traces):
    """It should truncate traces to the specified dimension."""
    traces, features = sample_traces
    for trace, expected in zip(traces, features[:, :2]):
        assert np.array_equal(extract_sizes(trace, dimension=2), expected)


def test_extract_sizes_3d(sample_traces):
    """It should work with matices of traces as well.
    """
    traces, features = sample_traces
    assert np.array_equal(extract_sizes_3d(traces, dimension=3),
                          features[:, :3])


def test_extract_sizes_3d_infer(sample_traces):
    """It should infer the dimension if not specified.
    """
    traces, features = sample_traces
    assert np.array_equal(extract_sizes_3d(traces), features)
