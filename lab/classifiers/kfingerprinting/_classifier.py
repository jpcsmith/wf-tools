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
)

import scipy.spatial.distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
)


Element = TypeVar('Element')

_LOGGER = logging.getLogger(__name__)


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
        self.forest = forest
        self.n_neighbours = n_neighbours
        self._graph: Optional[NearestNeighbors] = None
        self._labels = None

    def fit(self, X, y):  # pylint: disable=invalid-name
        """Fit the estimator according to the given training data."""
        self.forest.fit(X, y)
        self._graph = NearestNeighbors(
            n_neighbours=self.n_neighbours,
            metric=scipy.spatial.distance.hamming,
            n_jobs=self.forest.n_jobs)
        self._graph.fit(self.forest.apply(X))
        self._labels = y

    def predict(self, X):  # pylint: disable=invalid-name
        """Predict the class for X.

        The predicted class is the unanimous label of the k-closest neighbours
        or None.
        """
        assert self._graph is not None
        neighbourhoods = self._graph.kneighbors(X, return_distance=False)
        result = []
        for neighbours_list in neighbourhoods:
            labels = [self._labels[index] for index in neighbours_list]
            result.append(_unique_element(labels))
        return result


# @dataclass
# class Config:
#     """Classifier configuration."""
#     fg_train_size: Union[float, int] = 0.6
#     bg_train_size: Union[float, int] = 5000
#     num_trees: int = 1000
#
#
# # A dataset has 3 index columns, a bool column 'foreground', a unique index and
# # 'protocol' being 'quic' or 'tcp'
#
# def _stratified_split(dataset: pd.DataFrame, train_size: Union[float, int],
#                       random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Split the dataset into training and testing dataframes."""
#     splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
#                                       random_state=random_state)
#     train_index, test_index = next(
#         splitter.split(dataset.drop(columns='label'), dataset['label']))
#
#     return dataset[train_index], dataset[test_index]
#
#
# def openworld_leaves(dataset: pd.DataFrame, config: Optional[Config] = None,
#                      random_state: int = 42):
#     """Produces the leaf vectors used for classification."""
#     config = config or Config()
#
#
#
#     fg_train, fg_test = _stratified_split(fg_dataset, fg_train_size,
#                                           random_state)
#     _LOGGER.info("Split the foreground class into train and test sets of %d "
#                  "and %d rows respectively.", len(fg_train), len(fg_test))
#
#     bg_train, bg_test = _stratified_split(bg_dataset, bg_train_size,
#                                           random_state)
#     _LOGGER.info("Split the background class into train and test sets of %d "
#                  "and %d rows respectively.", len(bg_train), len(bg_test))
#
#     train = bg_train.append(fg_train, ignore_index=True)
#     test = bg_test.append(fg_test, ignore_index=True)
#
#     _LOGGER.info("Planting a random forest...")
#     model = RandomForestClassifier(n_estimators=n_trees, oob_score=True,
#                                    n_jobs=-1)
#     model.fit(train.drop(columns='label'), train['label'])
#     _LOGGER.info("Planting complete.")
#
#     train_leaf = pd.DataFrame(model.apply(train.drop(columns='label')),
#                               index=train.index)
#     train_leaf['label'] = train['label']
#     _LOGGER.debug("Training leaves %s", train_leaf)
#
#     test_leaf = pd.DataFrame(model.apply(test.drop(columns='label')),
#                              index=test.index)
#     test_leaf['label'] = test['label']
#     _LOGGER.debug("Testing leaves %s", test_leaf)
#
#     return train_leaf, test_leaf
