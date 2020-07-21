"""Tests for the KFingerprintingClassifier."""
import pytest
import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.model_selection import train_test_split
from sklearn import metrics

from lab.classifiers.kfingerprinting import (
    KFingerprintingClassifier, extract_features,
)


@parametrize_with_checks([KFingerprintingClassifier])
def test_sklearn_compatiblity(estimator, check):
    """Test that the k-fingerprinting classifier is compatible with sklearn."""
    if check.func.__name__ == 'check_supervised_y_2d':
        pytest.skip("KFingerprintingClassifier does not support 2d labels.")
    check(estimator)


@pytest.fixture(name="train_test_data")
def fixture_train_test_data(dataset) -> tuple:
    """Return a tuple of (x_train, x_test, y_train, y_test) in the
    open-world setting.
    """
    sizes, times, classes = dataset
    features = np.ndarray((len(sizes), 165), dtype=float)

    for i, (size_row, times_row) in enumerate(zip(sizes, times)):
        trace = np.recarray((len(size_row), ), dtype=[
            ("timestamp", "f8"), ("direction", "i1"), ("size", "i4")])
        trace["timestamp"] = times_row
        np.sign(size_row, out=trace["direction"])
        np.abs(size_row, out=trace["size"])
        features[i] = extract_features(trace)
    return train_test_split(
        features, classes, stratify=classes, random_state=202155)


def test_kfp_predict(train_test_data):
    """Sanity test of accurate predictions."""
    x_train, x_test, y_train, y_test = train_test_data

    classifier = KFingerprintingClassifier(random_state=2057)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    assert metrics.accuracy_score(y_test, predictions) > 0.8


def test_kfp_predict_proba(train_test_data):
    """Sanity test that we can predict with probabilities."""
    x_train, x_test, y_train, y_test = train_test_data
    n_classes = 3

    classifier = KFingerprintingClassifier(random_state=2057)
    classifier.fit(x_train, y_train)
    probabilities = classifier.predict_proba(x_test)
    assert probabilities.shape == (len(y_test), n_classes)

    predictions = classifier.classes_[np.argmax(probabilities, axis=1)]
    assert metrics.accuracy_score(y_test, predictions) > 0.8


def test_kfp_predict_unanimous(train_test_data):
    """Unanimous predictions should be equivalent to the class that has
    a score of 1.0
    """
    x_train, x_test, y_train, _ = train_test_data

    classifier = KFingerprintingClassifier(random_state=2057)
    classifier.fit(x_train, y_train)

    probabilities = classifier.predict_proba(x_test)
    # If all are false, then argmax will return 0, which happens to be the index
    # of the -1 label, thus mapping to -1 when for no unanimous (1.0) decision
    expected = classifier.classes_[np.argmax(probabilities == 1., axis=1)]

    np.testing.assert_array_equal(
        expected, classifier.predict_unanimous(x_test))
