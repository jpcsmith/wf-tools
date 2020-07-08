"""Tests for the classifiers of p1-FP."""
import pytest
import sklearn
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from lab.classifiers.p1fp import P1FPClassifierC
from lab.classifiers.p1fp._classifier import onehot


@pytest.mark.skip
@parametrize_with_checks([P1FPClassifierC])
def test_sklearn_compatiblity(estimator, check):
    """Test that the p1-FP(C) classifier is compatible with sklearn."""
    check(estimator)


@pytest.fixture(name="iris_samples")
def fixture_iris_samples() -> tuple:
    """Return (X_train, X_test, y_train, y_test) from the iris dataset.
    """
    X, y = sklearn.datasets.load_iris(return_X_y=True)
    return sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)


def test_one_hot_encoding(iris_samples):
    """Test the equivalence of the one hot encodings."""
    _, _, y_train, _ = iris_samples

    encoder = OneHotEncoder(sparse=False)
    assert np.array_equal(
        onehot(y_train, n_classes=3),
        encoder.fit_transform(y_train.reshape(-1, 1)))


def test_p1fpclassifierc_classify(iris_samples):
    """Test that it performs classification."""
    x_train, x_test, y_train, y_test = iris_samples
    classifier = P1FPClassifierC(n_epoch=2)
    classifier.fit(x_train, y_train)

    prediction = classifier.predict(x_test)
    assert prediction.shape == (y_test.size, )
    assert np.all(np.isin(np.unique(prediction), np.unique(y_train)))

    prob_prediction = classifier.predict_proba(x_test)
    assert prob_prediction.shape == (y_test.size, np.unique(y_test).size)
