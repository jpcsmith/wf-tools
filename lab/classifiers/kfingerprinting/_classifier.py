"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

The original can be found at https://github.com/jhayes14/k-FP.
"""
import logging
from typing import (
    Optional,
    Iterable,
    TypeVar,
    Any,
)

import pandas as pd
import scipy.spatial.distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)


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

    Utilises a RandomForestClassifier to gather leaf indices then predicts a
    class or indeterminate using k-nearest neighbour and the Hamming distance
    between the leaf indices of the training data and that of the test data.

    Parameters
    ----------
    forest :
        Specifies the random forest classifier to use as the underlying forest.

    n_neighbours :
        The number of nearest neighbours to use for the prediction.


    References
    ----------
    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.
    """
    def __init__(self, forest: RandomForestClassifier, n_neighbours: int = 3):
        self._logger = logging.getLogger(__name__)
        self.forest = forest
        self.n_neighbours = n_neighbours
        self._graph: Optional[NearestNeighbors] = None
        self._labels = None

    def fit(self, X, y):  # pylint: disable=invalid-name
        """Fit the estimator according to the given training data."""
        self._logger.info("Fitting the random forest on %d samples.", len(X))
        self.forest.fit(X, y)

        self._graph = NearestNeighbors(
            n_neighbors=self.n_neighbours,
            metric=scipy.spatial.distance.hamming,
            n_jobs=self.forest.n_jobs)
        self._logger.info("Fitting the nearest neighbor graph.")
        self._graph.fit(self.forest.apply(X))
        self._labels = pd.Series(y)
        self._logger.info("Model fitting complete.")

    def predict(self, X, unknown_label: Any = 'unknown'):  # pylint: disable=invalid-name
        """Predict the class for X.

        The predicted class is the unanimous label of the k-closest neighbours
        or None.
        """
        assert self._graph is not None
        assert self._labels is not None
        self._logger.debug("Determining leaves for the prediction.")
        leaves = self.forest.apply(X)
        self._logger.debug("Identifying neighbours of the leaves.")
        neighbourhoods = self._graph.kneighbors(leaves, return_distance=False)
        result = []
        self._logger.debug("Formulating decision.")
        for neighbours_list in neighbourhoods:
            # breakpoint()
            labels = [self._labels.iloc[index] for index in neighbours_list]
            prediction = _unique_element(labels)
            # Explicitly check for None since 0/False are valid predictions
            result.append(prediction if prediction is not None
                          else unknown_label)
        return result
