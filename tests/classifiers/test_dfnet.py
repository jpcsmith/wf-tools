"""Tests for the DeepFingerprinting classifier."""
import pytest
from sklearn.model_selection import train_test_split

from lab.classifiers.dfnet import DeepFingerprintingClassifier
from lab.feature_extraction.trace import extract_sizes


@pytest.fixture(name="train_test_data")
def fixture_train_test_data(dataset) -> tuple:
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
    assert classifier.score(x_test, y_test) > 0.8
