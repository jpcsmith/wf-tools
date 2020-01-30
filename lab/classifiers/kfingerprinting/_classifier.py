"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

The original can be found at https://github.com/jhayes14/k-FP.
"""
import logging
import warnings
from typing import Optional, Iterable, TypeVar, Union

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array
from sklearn.utils.multiclass import unique_labels

Element = TypeVar('Element')


def _unique_element(items: Iterable[Element]) -> Optional[Element]:
    """If all the elements in the iterable are the same, it is returned,
    otherwise returns None.
    """
    items = set(items)
    if len(items) == 1:
        return items.pop()
    return None


class KFingerprintingClassifier(BaseEstimator, ClassifierMixin):
    """k-fingerprinting website classifier.

    Utilises a RandomForestClassifier to gather leaf indices then
    predicts a class or indeterminate using k-nearest neighbour and the
    Hamming distance between the leaf indices of the training data and
    that of the test data.

    Does not currently support 2-dimensional labels.


    Parameters
    ----------
    forest :
        Specifies the random forest classifier to use as the underlying
        forest.  If absent, one is created with 150 trees and
        oob_scoring, as well as using the n_jobs and random_state
        provided.  If provided in addtition to n_jobs and random_state,
        they are ignored.

    n_neighbours :
        The number of nearest neighbours to use for the prediction.

    unknown_label :
        The label to use for unknown predictions.  This is by default
        -1.  A warning is issues if the type of the unknown label does
        not match the type of the labels.

    References
    ----------
    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(self, forest: Optional[RandomForestClassifier] = None,
                 n_neighbours: int = 2, n_jobs=None, random_state=None,
                 unknown_label: Union[str, int] = -1):
        self.forest = forest
        self.n_neighbours = n_neighbours
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.unknown_label = unknown_label

    def fit(self, X, y):  # pylint: disable=invalid-name
        """Fit the estimator according to the given training data."""
        X = check_array(X, accept_sparse=False, dtype=None)
        y = check_array(y, accept_sparse=False, ensure_2d=False, dtype=None)

        # Check that the unknown label is compatible with the other labels
        if y.dtype != np.dtype(type(self.unknown_label)):
            warnings.warn(("The datatype of the labels ({}) does not match the "
                           "datatype of the unknown label ({}). A conversion "
                           "or error may occur.").format(
                               y.dtype, np.dtype(type(self.unknown_label))))

        logger = logging.getLogger(__name__)
        logger.debug("Fitting the random forest on %d samples.", len(X))

        # pylint: disable=attribute-defined-outside-init
        self.classes_ = unique_labels(y)
        self.forest_ = (sklearn.base.clone(self.forest)
                        if self.forest is not None
                        else RandomForestClassifier(
                            n_estimators=150, oob_score=True,
                            n_jobs=self.n_jobs, random_state=self.random_state))

        self.forest_.fit(X, y)

        logger.debug("Fitting the nearest neighbor graph.")
        self.graph_ = NearestNeighbors(n_neighbors=self.n_neighbours,
                                       metric='hamming', n_jobs=self.n_jobs)

        self.graph_.fit(self.forest_.apply(X))

        # We need the labels because the graph returns indices into the
        # population matrix, which are the same indices associated with
        # the labeles
        assert isinstance(y, np.ndarray)
        self.labels_ = y

        logger.debug("Model fitting complete.")
        return self

    def predict(self, X, n_neighbors: Optional[int] = None):
        """Predict the class for X.

        The predicted class is the unanimous label of the k-closest neighbours
        or None.
        """
        sklearn.utils.validation.check_is_fitted(
            self, ['graph_', 'labels_', 'forest_'])

        logger = logging.getLogger(__name__)
        logger.debug("Determining leaves for the prediction.")

        leaves = self.forest_.apply(X)

        logger.debug("Identifying neighbours of the leaves.")
        neighbourhoods = self.graph_.kneighbors(
            leaves, return_distance=False, n_neighbors=n_neighbors)

        logger.debug("Formulating decision.")
        result = []
        for neighbours_list in neighbourhoods:
            labels = [self.labels_[index] for index in neighbours_list]
            prediction = _unique_element(labels)
            # Explicitly check for None since 0/False are valid predictions
            result.append(prediction if prediction is not None
                          else self.unknown_label)
        return np.array(result)
