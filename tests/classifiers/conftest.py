"""Shared fixtures for classifier tests."""
import pathlib
from typing import Tuple

import h5py
import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from lab.feature_extraction.trace import ensure_non_ragged


def pytest_addoption(parser):
    """Add options to pytest CLI."""
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """Update the list of available markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Modify the collected tests."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(name="dataset")
def fixture_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a tuple of (sizes, timestamps, labels), appropriate for
    testing the classifiers in the open-world setting.
    """
    data_path = pathlib.Path(__file__).with_name("test-dataset.hdf")

    with h5py.File(str(data_path), mode="r") as infile:
        labels = np.array(infile["/labels"])
        sizes = np.array(infile["/sizes"])
        timestamps = np.array(infile["/timestamps"])
    return (sizes, timestamps, labels)


@pytest.fixture(name="train_test_sizes")
def fixture_train_test_sizes(dataset) -> tuple:
    """Return a tuple of (x_train, x_test, y_train, y_test) in the
    open-world setting.
    """
    sizes, _, classes = dataset
    features = ensure_non_ragged(sizes)[:, :5000]
    assert len(np.unique(classes)) == 11
    assert features.shape[1] == 5000

    return train_test_split(
        features, classes, stratify=classes, random_state=7141845)
