"""Tests for the DeepFingerprinting classifier."""
import pytest
from lab.classifiers.dfnet import DeepFingerprintingClassifier


@pytest.mark.slow
def test_deep_fingerprinting(train_test_sizes):
    """Simple sanity test."""
    x_train, x_test, y_train, y_test = train_test_sizes

    classifier = DeepFingerprintingClassifier(
        n_features=5000, n_classes=3, epochs=10)

    classifier.fit(x_train, y_train)
    assert classifier.score(x_test, y_test) > 0.8
