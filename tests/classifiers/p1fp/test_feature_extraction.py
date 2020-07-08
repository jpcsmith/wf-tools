"""Tests for feature extraction."""
from typing import Tuple

import pytest
import numpy as np

from lab.classifiers.p1fp.feature_extraction import vectorize_traces


@pytest.fixture(name="sample_traces")
def fixture_sample_traces() -> Tuple[list, np.ndarray]:
    """Returns a simple list of traces."""
    traces = [
        [(0, 1, 1350), (0.01, 1, 1350), (0.02, -1, 600), (0.03, 1, 70)],
        [(0, 1, 1300), (0.015, 1, 1350), (0.025, 1, 1200)],
        [(0, 1, 1200), (0.02, 1, 1350)],
    ]
    expected_features = np.array([
        [1350, 1350, -600, 70],
        [1300, 1350, 1200, 0],
        [1200, 1350, 0, 0]
    ])
    return traces, expected_features


def test_vectorize_traces(sample_traces):
    """It should create features from the vectorized traces."""
    traces, features = sample_traces
    result = vectorize_traces(traces, n_features=features.shape[1])
    assert np.array_equal(result, features)


def test_vectorize_traces_truncate(sample_traces):
    """It should truncate long traces to the specified number of
    features.
    """
    traces, features = sample_traces
    result = vectorize_traces(traces, n_features=2)
    assert np.array_equal(result, features[:, :2])


def test_vectorize_pad(sample_traces):
    """It should pad short traces to the specified number of features.
    """
    traces, features = sample_traces
    result = vectorize_traces(traces, n_features=6)
    assert np.array_equal(result, np.concatenate(
        (features, np.zeros((features.shape[0], 2))), axis=1))
