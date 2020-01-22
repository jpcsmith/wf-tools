"""Combination and other classifiers."""
from typing import Sequence, Tuple, Any

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_array, validation
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils import multiclass


class ConditionalClassifier(_BaseComposition, sklearn.base.ClassifierMixin):
    """Performs classification using a distinguisher to first identify
    which of two classifiers to use.

    Parameters
    ----------
    estimators :
        A sequence of 1 to 3 estimators described below, described by
        (name, estimator). If a single estimator is provided in the
        sequence, then it is cloned and use for all three estimators.
        If two are provided, then the first is used as the distinguisher,
        and the second is used as both the positive and negative
        classifiers.  Otherwise, the third is used as the negative
        classifier.

        distinguisher :
            A binary classifier that will determine which of classifier_pos or
            classifier_neg to use.
        pos :
            Fitted on, and makes predictions for only those samples identified
            as pos_label.
        neg :
            Fitted on, and makes predictions for only those samples identified
            as not being pos_label.

    Note
    ----
    As we are using a multidimensional numpy array for the y-values, be careful
    as numpy arrays only store one type.  pos_label is currently an int, and
    will therefore not compare truthily to a pos_label which is a string, for
    example '1', which may occur if the class labels are strings.
    """

    _required_parameters = ['estimators']

    def __init__(self, estimators: Sequence[Tuple[str, Any]], pos_label=1):
        super().__init__()
        self.estimators = estimators
        self.pos_label = pos_label

    def _more_tags(self):
        return {'multioutput_only': True}

    def get_params(self, deep=True) -> dict:
        """Get parameters for this estimator."""
        return self._get_params('estimators', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator."""
        self._set_params('estimators', **kwargs)
        return self

    def _clone_estimators(self, clone: bool) -> Sequence[Tuple[str, int]]:
        from sklearn import base
        from itertools import chain, repeat

        assert 1 <= len(self.estimators) <= 3
        estimators = chain(self.estimators, repeat(self.estimators[-1]))
        return [
            (f'{name}_{tag}', base.clone(estimator) if clone else estimator)
            for tag, (name, estimator) in zip(['distinguisher', 'pos', 'neg'],
                                              estimators)
        ]

    @property
    def distinguisher_(self):
        """Return the distinguishing estimator."""
        validation.check_is_fitted(self, ['estimators_'])
        return self.estimators_[0][1]

    @property
    def pos_classifier_(self):
        """Return the classifier used for the positive samples."""
        validation.check_is_fitted(self, ['estimators_'])
        return self.estimators_[1][1]

    @property
    def neg_classifier_(self):
        """Return the classifier used for the negative samples."""
        validation.check_is_fitted(self, ['estimators_'])
        return self.estimators_[2][1]

    def fit(self, X, y, clone: bool = True) -> None:
        """Fit the distinguishers and sub-classifiers on the provided data.

        Parameters
        ----------
        y :
            A 2d-array where the first column has binary predictions to fit
            the distinguisher.
        """
        X = check_array(X, accept_sparse=False, dtype='numeric')
        y = check_array(y, accept_sparse=False, ensure_2d=True, dtype='numeric')
        multiclass.check_classification_targets(y)
        assert multiclass.type_of_target(y[:, 0]) == 'binary'

        # pylint: disable=attribute-defined-outside-init
        self.estimators_ = self._clone_estimators(clone)
        self.classes_ = np.unique(y, axis=0)

        if self.pos_label not in multiclass.unique_labels(y[:, 0]):
            raise ValueError("pos_label={!r} is not a valid label: {!r}".format(
                self.pos_label, multiclass.unique_labels(y[:, 0])))

        self.distinguisher_.fit(X, y[:, 0])

        mask = y[:, 0] == self.pos_label
        self.pos_classifier_.fit(X[mask], y[mask, 1])
        self.neg_classifier_.fit(X[~mask], y[~mask, 1])

    def predict(self, X) -> np.ndarray:
        """Distinguish and predict the class using the appropriate classifer.

        Return an array of size (n_samples, 2).
        """
        validation.check_is_fitted(self, ['estimators_', 'classes_'])

        X = check_array(X, accept_sparse=False, dtype=None)

        choice = np.asarray(self.distinguisher_.predict(X))
        mask = choice == self.pos_label

        pos_predictions = np.array(self.pos_classifier_.predict(X[mask]))
        neg_predictions = np.array(self.neg_classifier_.predict(X[~mask]))

        assert pos_predictions.dtype == neg_predictions.dtype
        predictions = np.ndarray(len(X), dtype=pos_predictions.dtype)
        predictions[mask] = pos_predictions
        predictions[~mask] = neg_predictions

        assert predictions.dtype == choice.dtype
        return np.array(list(zip(choice, predictions)), dtype=choice.dtype)
