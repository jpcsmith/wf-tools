"""Module containing shared code for analysis."""
import time
import logging
from dataclasses import dataclass
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from sklearn.ensemble import RandomForestClassifier

from lab.classifiers.kfingerprinting import (
    KFingerprintingClassifier,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    """Classifier configuration."""
    fg_train_size: Union[float, int] = 0.6
    bg_train_size: Union[float, int] = 0.05

    n_trees: int = 1000
    n_neighbors: int = 3
    n_jobs: int = -1
    random_state: Optional[np.random.RandomState] = None


def train(features, labels, config: Config) -> KFingerprintingClassifier:
    """Train a kFingerpriting model and return it."""
    model = KFingerprintingClassifier(
        RandomForestClassifier(
            n_estimators=config.n_trees, oob_score=True,
            n_jobs=config.n_jobs, random_state=config.random_state),
        n_neighbours=config.n_neighbors)

    _LOGGER.info("Training the classifier with %d samples and %d trees.",
                 len(labels), config.n_trees)

    start = time.perf_counter()
    model.fit(features, labels)
    end = time.perf_counter()
    _LOGGER.info("Training complete in %.2fs", (end-start))

    return model


def predict(model: KFingerprintingClassifier, unlabelled_data, unknown_label,
            n_neighbors=None) -> np.ndarray:
    """Return the predicted labels using the model."""
    start = time.perf_counter()
    predictions = model.predict(unlabelled_data, unknown_label=unknown_label,
                                n_neighbors=n_neighbors)
    end = time.perf_counter()
    _LOGGER.info("Prediction complete in %.2fs", (end - start))
    return predictions


def to_ndarrays(dataset: pd.DataFrame, protocol: Optional[str] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Convert the dataset to a simple np.ndarrays and return as
    (features, labels).

    Will filter the dataset based the protocol if provided.
    """
    assert protocol != ''
    if protocol is not None:
        dataset = dataset.loc[idx[:, protocol], ]
    dataset = dataset.sort_index(level='label')
    labels = pd.Series(
        dataset.index.get_level_values('label').values, dtype='category')
    return (dataset.values, labels.cat.codes.values)


def sample_labels(dataset: pd.DataFrame, sample_size: int,
                  random_state: np.random.RandomState):
    """Return a subset of the labels."""
    labels = dataset.index.get_level_values('label').unique().values
    choices = random_state.choice(labels, sample_size, replace=False)
    mask = dataset.index.get_level_values('label').isin(choices)
    return dataset[mask]
