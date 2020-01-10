"""Tests for lab.classifiers.other classifiers."""
from unittest.mock import Mock

import pytest
import numpy as np
from sklearn.dummy import DummyClassifier

from lab.classifiers.other import ConditionalClassifier


@pytest.fixture(name='mocked_cond_classifier')
def fixture_mocked_cond_classifier() -> ConditionalClassifier:
    """Return an instance of the Conditional classifier with its components
    mocked.
    """
    classifier = ConditionalClassifier(
        DummyClassifier(), DummyClassifier(), DummyClassifier())
    classifier.distinguisher = Mock(name='MockDistinguisher')
    classifier.classifier_pos = Mock(name='MockClassifierA')
    classifier.classifier_neg = Mock(name='MockClassifierB')
    return classifier


@pytest.fixture(name='test_dataset')
def fixture_test_dataset():
    """Return a feature array of size (9, 3) and class labels of type
    (9, 2).

    There are 3 classes, 0, 1 and 2.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15],
                  [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]])
    y = np.array([[1, 0], [1, 0], [-1, 0],
                  [1, 1], [-1, 1], [-1, 1],
                  [1, 2], [1, 2], [-1, 2]])
    pos_idx = [0, 1, 3, 6, 7]
    neg_idx = [2, 4, 5, 8]
    return X, y, pos_idx, neg_idx


def test_conditional_classifier_fit(mocked_cond_classifier):
    """Test that the fit method passes the correct arguments to the
    sub-classifiers
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([[1, 0], [-1, 0], [1, 1]])

    mocked_cond_classifier.fit(X, y)

    mocked_cond_classifier.distinguisher.fit.assert_called_once()
    test_x, test_y = mocked_cond_classifier.distinguisher.fit.call_args[0]
    np.testing.assert_array_equal(X, test_x)
    np.testing.assert_array_equal([1, -1, 1], test_y)

    mocked_cond_classifier.classifier_pos.fit.assert_called_once()
    test_x, test_y = mocked_cond_classifier.classifier_pos.fit.call_args[0]
    np.testing.assert_array_equal([[1, 2, 3], [7, 8, 9]], test_x)
    np.testing.assert_array_equal([0, 1], test_y)

    mocked_cond_classifier.classifier_neg.fit.assert_called_once()
    test_x, test_y = mocked_cond_classifier.classifier_neg.fit.call_args[0]
    np.testing.assert_array_equal([[4, 5, 6]], test_x)
    np.testing.assert_array_equal([0], test_y)


# def test_hard_predict_proba(mocked_cond_classifier, test_dataset):
#     """It should correctly combine the predicitions."""
#     X, y = test_dataset
#     classifier = mocked_cond_classifier
#
#     classifier.distinguisher.predict_proba.return_value = np.array(
#         [0.6, 0.7, 0.1, 0.8, 0.2, 0.3, 0.9, 0.6, 0.4])
#     classifier.classifier_pos.predict.return_value = np.array(
#         [[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1],
#          [0.05, 0.9, 0.05], [0.2, 0.2, 0.6], [0.2, 0.1, 0.7]])
#     classifier.classifier_neg.predict.return_value = np.array(
#         [[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.2, 0.8, 0], [0.1, 0.0, 0.9]])
#
#     test_y = classifier.predict_proba(X)
#
#     classifier.distinguisher.predict.assert_called_once()
#     test_x = classifier.distinguisher.predict.call_args[0][0]
#     np.testing.assert_array_equal(X, test_x)
#
#     classifier.classifier_pos.predict.assert_called_once()
#     test_x = classifier.classifier_pos.predict.call_args[0][0]
#     np.testing.assert_array_equal(
#         [[1, 2, 3], [4, 5, 6], [10, 11, 12], [19, 20, 21], [22, 23, 24]],
#         test_x)
#
#     classifier.classifier_neg.predict.assert_called_once()
#     test_x = classifier.classifier_neg.predict.call_args[0][0]
#     np.testing.assert_array_equal(
#         [[7, 8, 9], [13, 14, 15], [16, 17, 18], [25, 26, 27]], test_x)
#
#     np.testing.assert_array_equal(y, test_y)
#
#
#         [0.6, 0.7, 0.1, 0.8, 0.2, 0.3, 0.9, 0.6, 0.4])
#     classifier.classifier_pos.predict.return_value = np.array(
#         [[0.6, 0.3, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1],
#          [0.05, 0.9, 0.05], [0.2, 0.2, 0.6], [0.2, 0.1, 0.7]])
#     classifier.classifier_neg.predict.return_value = np.array(
#         [[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.2, 0.8, 0], [0.1, 0.0, 0.9]])


def test_hard_predict(mocked_cond_classifier, test_dataset):
    """It should correctly combine the predicitions."""
    X, y, pos_idx, neg_idx = test_dataset
    classifier = mocked_cond_classifier

    classifier.distinguisher.predict.return_value = y[:, 0]
    classifier.classifier_pos.predict.return_value = y[pos_idx, 1]
    classifier.classifier_neg.predict.return_value = y[neg_idx, 1]

    test_y = classifier.predict(X)

    classifier.distinguisher.predict.assert_called_once()
    test_x = classifier.distinguisher.predict.call_args[0][0]
    np.testing.assert_array_equal(X, test_x)

    classifier.classifier_pos.predict.assert_called_once()
    test_x = classifier.classifier_pos.predict.call_args[0][0]
    np.testing.assert_array_equal(X[pos_idx], test_x)

    classifier.classifier_neg.predict.assert_called_once()
    test_x = classifier.classifier_neg.predict.call_args[0][0]
    np.testing.assert_array_equal(X[neg_idx], test_x)

    np.testing.assert_array_equal(y, test_y)
