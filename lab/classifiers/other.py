"""Combination and other classifiers."""
from typing import TypeVar

import numpy as np
import sklearn


Estimator = TypeVar('Estimator')


class ConditionalClassifier(sklearn.base.BaseEstimator):
    """

    Parameters
    ----------
    distinguisher :
        A binary classifier that will determine which of classifier_pos or
        classifier_neg to use.
    classifier_pos :
        Fitted on, and makes predictions for only those samples identified
        as pos_label.
    classifier_neg :
        Fitted on, and makes predictions for only those samples identified
        as not being pos_label.
    voting :
        Either 'hard' or 'soft'.  If 'soft' the decision is made based on
        the probabilites of the distinguisher and classifier, and thus they
        must support the predict_proba method.

    Note
    ----
    As we are using a multidimensional numpy array for the y-values, be careful
    as numpy arrays only store one time.  pos_label is currently an int, and
    will therefore not compare truthily to a pos_label which is a string, for
    example '1', which may occur if the class labels are strings.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, distinguisher: Estimator, classifier_pos: Estimator,
                 classifier_neg: Estimator, pos_label=1, voting: str = 'hard'):
        self.distinguisher = sklearn.base.clone(distinguisher)
        self.classifier_pos = sklearn.base.clone(classifier_pos)
        self.classifier_neg = sklearn.base.clone(classifier_neg)
        self.pos_label = pos_label

        assert voting in ('hard', 'soft')
        self.voting = voting

    def fit(self, X, y: np.ndarray) -> None:
        """Fit the distinguishers and sub-classifiers on the provided data.

        Parameters
        ----------
        y :
            A 2d-array where the first column has binary predictions to fit
            the distinguisher.
        """
        assert self.pos_label in y[:, 0]
        self.distinguisher.fit(X, y[:, 0])

        mask = y[:, 0] == self.pos_label
        self.classifier_pos.fit(X[mask], y[mask, 1])
        self.classifier_neg.fit(X[~mask], y[~mask, 1])

    def predict_soft(self, X) -> np.ndarray:
        """Return the argmax of the probability predictions, weighted according
        to the distinguisher.

        This is invoked by predict when the voting type is soft.
        """

    def predict(self, X) -> np.ndarray:
        """Distinguish and predict the class using the appropriate classifer.

        Return an array of size (n_samples, 2).
        """
        if self.voting == 'soft':
            return self.predict_soft(X)

        choice = np.array(self.distinguisher.predict(X))
        mask = choice == self.pos_label

        pos_predictions = np.array(self.classifier_pos.predict(X[mask]))
        neg_predictions = np.array(self.classifier_neg.predict(X[~mask]))

        assert pos_predictions.dtype == neg_predictions.dtype
        predictions = np.ndarray(len(X), dtype=pos_predictions.dtype)
        predictions[mask] = pos_predictions
        predictions[~mask] = neg_predictions

        assert predictions.dtype == choice.dtype
        return np.array(list(zip(choice, predictions)))

    def predict_proba(self, X) -> np.ndarray:
        """Predict the weights for the various classes."""
