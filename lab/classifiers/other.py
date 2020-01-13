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

        Fixes 1 for the positive class and -1 for the negative class.
        """
        probs = self.predict_proba_soft(X)
        mask = probs[:, 0] > 0.5

        result = np.ndarray((len(probs), 2), dtype=int)
        result[:, 0] = np.where(mask, 1, -1)
        result[:, 1] = np.argmax(probs[:, 1:], axis=1)
        return result

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

    def predict_proba_soft(self, X) -> np.ndarray:
        """Return the probability predictions, weighted according to the
        distinguisher.

        This is invoked by predict_proba when the voting type is soft.
        """
        choice_proba = np.array(self.distinguisher.predict_proba(X))
        choice_proba = choice_proba.reshape(len(choice_proba), 1)

        pos_predictions = np.array(self.classifier_pos.predict_proba(X))
        neg_predictions = np.array(self.classifier_neg.predict_proba(X))
        n_classes = pos_predictions.shape[1]  # pylint: disable=unsubscriptable-object

        predictions = (choice_proba * pos_predictions) + (
            (1 - choice_proba) * neg_predictions)

        result = np.ndarray((len(X), n_classes + 1), dtype=float)
        result[:, 0] = choice_proba[:, 0]
        result[:, 1:] = predictions

        return result

    def predict_proba(self, X) -> np.ndarray:
        """Return the belief in the decision as well as in that of the various
        classes.

        The resulting ndarray is of shape (n_samples, n_classes + 1), where the
        first column specifies the belief in the decision, and the remaining
        specify the beliefs in the classes.
        """
        if self.voting == 'soft':
            return self.predict_proba_soft(X)

        choice_proba = np.array(self.distinguisher.predict_proba(X))
        mask = choice_proba > 0.5

        pos_predictions = np.array(self.classifier_pos.predict_proba(X[mask]))
        neg_predictions = np.array(self.classifier_neg.predict_proba(X[~mask]))

        assert pos_predictions.dtype == neg_predictions.dtype \
            == choice_proba.dtype
        # pylint: disable=unsubscriptable-object
        assert pos_predictions.shape[1] == neg_predictions.shape[1]
        n_classes = pos_predictions.shape[1]
        # pylint: enable=unsubscriptable-object

        result = np.ndarray((len(X), n_classes + 1), dtype=float)
        result[mask, 1:] = pos_predictions
        result[~mask, 1:] = neg_predictions
        result[:, 0] = choice_proba

        return result
