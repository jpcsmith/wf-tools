"""Tests for lab.classifiers.other classifiers."""
import unittest
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


def test_hard_predict_proba(mocked_cond_classifier, test_dataset):
    """It should correctly combine the predicitions, with the decision
    probability in the first column.
    """
    X, _, pos_idx, neg_idx = test_dataset
    classifier = mocked_cond_classifier

    y_probs = np.array(
        [[0.6, 0.6, 0.3, 0.1], [0.7, 0.7, 0.2, 0.1], [0.1, 0.8, 0.1, 0.1],
         [0.8, 0.1, 0.8, 0.1], [0.2, 0.1, 0.7, 0.2], [0.3, 0.2, 0.8, 0],
         [0.9, 0, 0.1, 0.9], [0.6, 0.1, 0.1, 0.8], [0.4, 0.2, 0.1, 0.7]])

    classifier.distinguisher.predict_proba.return_value = y_probs[:, 0]
    classifier.classifier_pos.predict_proba.return_value = y_probs[pos_idx, 1:4]
    classifier.classifier_neg.predict_proba.return_value = y_probs[neg_idx, 1:4]

    test_y = classifier.predict_proba(X)

    classifier.distinguisher.predict_proba.assert_called_once()
    test_x = classifier.distinguisher.predict_proba.call_args[0][0]
    np.testing.assert_array_equal(X, test_x)

    classifier.classifier_pos.predict_proba.assert_called_once()
    test_x = classifier.classifier_pos.predict_proba.call_args[0][0]
    np.testing.assert_array_equal(X[pos_idx], test_x)

    classifier.classifier_neg.predict_proba.assert_called_once()
    test_x = classifier.classifier_neg.predict_proba.call_args[0][0]
    np.testing.assert_array_equal(X[neg_idx], test_x)

    np.testing.assert_array_equal(y_probs, test_y)


@pytest.fixture(name='class_cond_classifier', scope='function')
def fixture_class_cond_classifier(request, mocked_cond_classifier):
    """Enable the use of the mocked_cond_classifier in unittest classes.
    """
    request.cls.mocked_cond_classifier = mocked_cond_classifier


@pytest.mark.usefixtures('class_cond_classifier')
class SoftPredictionsTest(unittest.TestCase):
    """Test fixture for the soft predictions."""
    def setUp(self):
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.classifier = self.mocked_cond_classifier  # pylint: disable=no-member
        self.classifier.voting = 'soft'

        self.pos_predictions = np.array([[0.6, 0.3, 0.1],
                                         [0.5, 0.2, 0.3],
                                         [0.8, 0.1, 0.1]])
        self.neg_predictions = np.array([[0.5, 0.4, 0.1],
                                         [0.4, 0.1, 0.5],
                                         [0.2, 0.2, 0.6]])
        self.y_probs = np.array([[0.5, 0.55, 0.35, 0.1],
                                 [0.1, 0.41, 0.11, 0.48],
                                 [0.4, 0.44, 0.16, 0.4]])
        self.y = np.array([[-1, 0], [-1, 2], [-1, 0]])

    def test_soft_predict_proba(self):
        """It should correctly combine the predicitions, with the decision
        probability in the first column.
        """
        self.classifier.distinguisher.predict_proba.return_value = \
            self.y_probs[:, 0]
        self.classifier.classifier_pos.predict_proba.return_value = \
            self.pos_predictions
        self.classifier.classifier_neg.predict_proba.return_value = \
            self.neg_predictions

        test_y = self.classifier.predict_proba(self.X)

        self.classifier.distinguisher.predict_proba.assert_called_once()
        test_x = self.classifier.distinguisher.predict_proba.call_args[0][0]
        np.testing.assert_array_equal(self.X, test_x)

        self.classifier.classifier_pos.predict_proba.assert_called_once()
        test_x = self.classifier.classifier_pos.predict_proba.call_args[0][0]
        np.testing.assert_array_equal(self.X, test_x)

        self.classifier.classifier_neg.predict_proba.assert_called_once()
        test_x = self.classifier.classifier_neg.predict_proba.call_args[0][0]
        np.testing.assert_array_equal(self.X, test_x)

        np.testing.assert_allclose(self.y_probs, test_y)

    def test_soft_predict(self):
        """It should correctly conclude whether the sample is pos or negative
        and claim a class.
        """
        self.classifier.predict_proba_soft = Mock()
        self.classifier.predict_proba_soft.return_value = self.y_probs

        test_y = self.classifier.predict(self.X)

        self.classifier.predict_proba_soft.assert_called_once()
        test_x = self.classifier.predict_proba_soft.call_args[0][0]
        np.testing.assert_array_equal(self.X, test_x)

        np.testing.assert_allclose(self.y, test_y)
