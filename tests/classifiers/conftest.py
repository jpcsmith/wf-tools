"""Shared fixtures for classifier tests."""
import pathlib
from typing import Tuple

import h5py
import pytest
import numpy as np


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
