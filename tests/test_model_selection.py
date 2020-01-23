"""Tests for the model selection code."""
import numpy as np
from numpy.random import RandomState
import sklearn.datasets
from sklearn.model_selection import StratifiedShuffleSplit

from lab.model_selection import SplitUnion


def test_split_union_sizes():
    """The split sizes should be correct."""
    # Each class has 50 samples
    X, y = sklearn.datasets.load_iris(return_X_y=True)

    rng = RandomState(42)
    splitter = SplitUnion(splitters=[
        ('zero', StratifiedShuffleSplit(3, train_size=0.4, random_state=rng)),
        ('onetwo', StratifiedShuffleSplit(3, train_size=10, random_state=rng))
    ])

    previous_indices = []
    for train_idx, test_idx in splitter.split(
            X, y, masks=[('zero', (y == 0)), ('onetwo', (y != 0))]):
        train_counts = np.bincount(y[train_idx])
        assert train_counts[0] == 20
        assert train_counts[1] == train_counts[2] == 5

        test_counts = np.bincount(y[test_idx])
        assert test_counts[0] == 30
        assert test_counts[1] == test_counts[2] == 45

        for prev_train, prev_test in previous_indices:
            assert set(train_idx) != set(prev_train)
            assert set(test_idx) != set(prev_test)
        previous_indices.append((train_idx, test_idx))
