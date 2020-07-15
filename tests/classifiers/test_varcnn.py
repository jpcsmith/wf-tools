"""Tests for the DeepFingerprinting classifier."""
import pytest
from sklearn.model_selection import train_test_split

import numpy as np
from lab.classifiers.varcnn import VarCNNClassifier
from lab.feature_extraction.trace import (
    ensure_non_ragged, extract_metadata, Metadata, extract_interarrival_times
)


@pytest.fixture(name="train_test_data", params=["sizes", "timestamps"])
def fixture_train_test_data(request, dataset) -> tuple:
    """Return a tuple of (x_train, x_test, y_train, y_test) in the
    closed-world setting.
    """
    sizes, times, classes = dataset
    assert len(np.unique(classes)) == 11

    if request.param == "sizes":
        main_features = ensure_non_ragged(sizes)[:, :5000]
    elif request.param == "timestamps":
        main_features = extract_interarrival_times(times)[:, :5000]
    else:
        raise ValueError(f"Unknown param {request.param}")
    assert main_features.shape[1] == 5000

    metadata = (Metadata.COUNT_METADATA | Metadata.TIME_METADATA
                | Metadata.SIZE_METADATA)
    metadata_features = extract_metadata(
        sizes=sizes, timestamps=times, metadata=metadata)
    assert metadata_features.shape[1] == 12

    features = np.hstack((main_features, metadata_features))

    return train_test_split(
        features, classes, stratify=classes, random_state=7152217)


@pytest.mark.slow
def test_varcnn_size(train_test_data):
    """Simple sanity test on the size features."""
    x_train, x_test, y_train, y_test = train_test_data

    classifier = VarCNNClassifier(
        n_classes=11, n_packet_features=5000, n_meta_features=12,
        epochs=20)
    classifier.fit(x_train, y_train, validation_split=0.2)
    assert classifier.score(x_test, y_test) > 0.8
