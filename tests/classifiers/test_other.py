"""Tests for lab.classifiers.other classifiers."""
from unittest.mock import Mock

import pytest
import numpy as np

from lab.classifiers.other import ConditionalClassifier


@pytest.fixture(name='mocked_cond_classifier')
def fixture_mocked_cond_classifier() -> ConditionalClassifier:
    """Return an instance of the Conditional classifier with its components
    mocked.
    """
    return ConditionalClassifier(
        [('a', Mock(name='Distinguisher')), ('b', Mock(name='Positive')),
         ('c', Mock(name='Negative'))])


@pytest.fixture(name='test_dataset')
def fixture_test_dataset():
    """Return a feature array of size (9, 3) and class labels of type
    (9, 2).

    There are 3 classes, 100, 200 and 300.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
                  [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]])
    y = np.array([[1, 100], [1, 100], [-1, 100],
                  [1, 200], [-1, 200], [-1, 200],
                  [1, 300], [1, 300], [-1, 300]])
    pos_idx = [0, 1, 3, 6, 7]
    neg_idx = [2, 4, 5, 8]
    return X, y, pos_idx, neg_idx


def test_conditional_classifier_fit(mocked_cond_classifier):
    """Test that the fit method passes the correct arguments to the
    sub-classifiers
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([[1, 100], [-1, 100], [1, 200]])
    classifier = mocked_cond_classifier

    classifier.fit(X, y, clone=False)

    classifier.distinguisher_.fit.assert_called_once()
    test_x, test_y = classifier.distinguisher_.fit.call_args[0]
    np.testing.assert_array_equal(X, test_x)
    np.testing.assert_array_equal([1, -1, 1], test_y)

    classifier.pos_classifier_.fit.assert_called_once()
    test_x, test_y = classifier.pos_classifier_.fit.call_args[0]
    np.testing.assert_array_equal([[1, 2, 3], [7, 8, 9]], test_x)
    np.testing.assert_array_equal([100, 200], test_y)

    classifier.neg_classifier_.fit.assert_called_once()
    test_x, test_y = classifier.neg_classifier_.fit.call_args[0]
    np.testing.assert_array_equal([[4, 5, 6]], test_x)
    np.testing.assert_array_equal([100], test_y)

    np.testing.assert_array_equal(
        classifier.classes_, [[-1, 100], [1, 100], [1, 200]])


def test_hard_predict(mocked_cond_classifier, test_dataset):
    """It should correctly combine the predicitions."""
    X, y, pos_idx, neg_idx = test_dataset
    classifier = mocked_cond_classifier

    # Just call fit so that it initialises what needs to be initialised
    classifier.fit(X, y, clone=False)

    classifier.distinguisher_.predict.return_value = y[:, 0]
    classifier.pos_classifier_.predict.return_value = y[pos_idx, 1]
    classifier.neg_classifier_.predict.return_value = y[neg_idx, 1]

    test_y = classifier.predict(X)

    classifier.distinguisher_.predict.assert_called_once()
    test_x = classifier.distinguisher_.predict.call_args[0][0]
    np.testing.assert_array_equal(X, test_x)

    classifier.pos_classifier_.predict.assert_called_once()
    test_x = classifier.pos_classifier_.predict.call_args[0][0]
    np.testing.assert_array_equal(X[pos_idx], test_x)

    classifier.neg_classifier_.predict.assert_called_once()
    test_x = classifier.neg_classifier_.predict.call_args[0][0]
    np.testing.assert_array_equal(X[neg_idx], test_x)

    np.testing.assert_array_equal(y, test_y)
