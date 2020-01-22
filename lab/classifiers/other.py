"""Combination and other classifiers."""
import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn import metrics


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

    Note
    ----
    As we are using a multidimensional numpy array for the y-values, be careful
    as numpy arrays only store one type.  pos_label is currently an int, and
    will therefore not compare truthily to a pos_label which is a string, for
    example '1', which may occur if the class labels are strings.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, distinguisher, classifier_pos, classifier_neg,
                 pos_label=1):
        self.distinguisher = distinguisher
        self.classifier_pos = classifier_pos
        self.classifier_neg = classifier_neg
        self.pos_label = pos_label
        self.encoder = sklearn.preprocessing.LabelEncoder()

    @property
    def classes_(self) -> np.ndarray:
        """The classes excluding the the distinguisher."""
        return self.encoder.classes_

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

        self.encoder.fit(y[:, 1])
        mask = y[:, 0] == self.pos_label
        self.classifier_pos.fit(X[mask], y[mask, 1])
        self.classifier_neg.fit(X[~mask], y[~mask, 1])

    def predict(self, X) -> np.ndarray:
        """Distinguish and predict the class using the appropriate classifer.

        Return an array of size (n_samples, 2).
        """
        choice = np.array(self.distinguisher.predict(X))
        mask = choice == self.pos_label

        pos_predictions = np.array(self.classifier_pos.predict(X[mask]))
        neg_predictions = np.array(self.classifier_neg.predict(X[~mask]))

        assert pos_predictions.dtype == neg_predictions.dtype
        predictions = np.ndarray(len(X), dtype=pos_predictions.dtype)
        predictions[mask] = pos_predictions
        predictions[~mask] = neg_predictions

        assert predictions.dtype == choice.dtype
        return np.array(list(zip(choice, predictions)), dtype=choice.dtype)

    def predict_proba(self, X) -> np.ndarray:
        """Return the belief in the decision as well as in that of the various
        classes.

        The resulting ndarray is of shape (n_samples, n_classes + 1), where the
        first column specifies the belief in the decision, and the remaining
        specify the beliefs in the classes.
        """
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

    def score(self, X, y, sample_weight=None) -> float:
        """Return the mean accuracy of the given test data and labels.

        The score for the conditional classifier only considers the final
        labels, and not the intermediate distinguishing labels.
        """
        return metrics.accuracy_score(
            y[:, 1], self.predict(X)[:, 1], sample_weight=sample_weight)
