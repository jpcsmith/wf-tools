"""Tests for the DeepFingerprinting classifier."""
import pathlib
from typing import Tuple

import h5py
import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from lab.classifiers.dfnet import DeepFingerprintingClassifier
from lab.feature_extraction.trace import extract_sizes


@pytest.fixture(name="dataset")
def fixture_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Return a tuple of (features, classes), appropriate for testing
    the DF classifier in the closed-world setting.
    """
    data_path = pathlib.Path(__file__).with_name("test-dataset.hdf")

    with h5py.File(str(data_path), mode="r") as infile:
        labels = infile["/monitored/labels"]
        # TODO: Use a dataset with only TCP and a single region
        mask = (labels["protocol"] == b"tcp")
        classes = labels[mask]["class"].copy()

        traces = infile["/monitored/traces"][np.nonzero(mask)[0], :5000].copy()
    return traces, classes


@pytest.fixture(name="train_test_data")
def fixture_train_test_data(dataset):
    """Return a tuple of (x_train, x_test, y_train, y_test) in the
    closed-world setting.
    """
    traces, classes = dataset
    features = extract_sizes(traces)
    return train_test_split(features, classes, random_state=7141845)


def test_df_on_sample(train_test_data):
    """Simple sanity test."""
    x_train, x_test, y_train, y_test = train_test_data

    classifier = DeepFingerprintingClassifier(
        n_features=5000, n_classes=10, epochs=1)
    classifier.fit(x_train, y_train)
    breakpoint()
    assert classifier.score(x_test, y_test) > 0.8
