"""An implementation of the k-fingerprinting classifier from:

    Hayes, Jamie, and George Danezis. "k-fingerprinting: A robust
    scalable website fingerprinting technique." 25th {USENIX} Security
    Symposium ({USENIX} Security 16). 2016.

The original can be found at https://github.com/jhayes14/k-FP.
"""
import logging
from typing import (
    Tuple,
    Union,
)

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier


_LOGGER = logging.getLogger(__name__)


def _stratified_split(dataset: pd.DataFrame, train_size: Union[float, int],
                      random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into training and testing dataframes."""
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size,
                                      random_state=random_state)
    train_index, test_index = next(
        splitter.split(dataset.drop(columns='label'), dataset['label']))

    return dataset[train_index], dataset[test_index]


def openworld_leaves(fg_dataset: pd.DataFrame, bg_dataset: pd.DataFrame,
                     fg_train_size: float = 0.6, bg_train_size: int = 5000,
                     n_trees: int = 1000, random_state: int = 42):
    """Produces the leaf vectors used for classification."""
    fg_train, fg_test = _stratified_split(fg_dataset, fg_train_size,
                                          random_state)
    _LOGGER.info("Split the foreground class into train and test sets of %d "
                 "and %d rows respectively.", len(fg_train), len(fg_test))

    bg_train, bg_test = _stratified_split(bg_dataset, bg_train_size,
                                          random_state)
    _LOGGER.info("Split the background class into train and test sets of %d "
                 "and %d rows respectively.", len(bg_train), len(bg_test))

    train = bg_train.append(fg_train, ignore_index=True)
    test = bg_test.append(fg_test, ignore_index=True)

    _LOGGER.info("Planting a random forest...")
    model = RandomForestClassifier(n_estimators=n_trees, oob_score=True,
                                   n_jobs=-1)
    model.fit(train.drop(columns='label'), train['label'])
    _LOGGER.info("Planting complete.")

    train_leaf = pd.DataFrame(model.apply(train.drop(columns='label')),
                              index=train.index)
    train_leaf['label'] = train['label']
    _LOGGER.debug("Training leaves %s", train_leaf)

    test_leaf = pd.DataFrame(model.apply(test.drop(columns='label')),
                             index=test.index)
    test_leaf['label'] = test['label']
    _LOGGER.debug("Testing leaves %s", test_leaf)

    return train_leaf, test_leaf


def distances(fg_dataset: pd.DataFrame, bg_dataset: pd.DataFrame,
              keep_top: int = 100):
    """Calculate distance from test instances between each training instance
    (which are used as labels). Default keeps the top 100 instances closest to
    the instance we are testing.
    """
    train_leaf, test_leaf = openworld_leaves(fg_dataset, bg_dataset)
