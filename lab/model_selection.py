"""Helpers for model selection."""
from typing import Sequence, Any, Tuple, List

import numpy as np
from sklearn.utils import Bunch


class SplitUnion:
    """Split subsets of a dataset with different classifiers, and
    remap the index to the entire dataset.

    splitters is a sequence of (str, CV splitter).  Return as many splits
    as the shortest splitter.
    """
    def __init__(self, splitters: Sequence[Tuple[str, Any]]):
        self.splitters = splitters

    @property
    def named_splitters(self) -> Bunch:
        """The splitters accessible by name."""
        return Bunch(**dict(self.splitters))

    def split(self, X, y, masks: Sequence[Tuple]):
        """Split X and y returning train, test indices.

        masks should be a sequence of (name, array mask), the same
        length as the splitters.  The mask is applied to the X and y
        arrays and passed to the splitter.  The masks should not
        overlap.
        """
        assert len(masks) == len(self.splitters)

        iters = {}
        for name, mask in masks:
            iters[name] = self.named_splitters[name].split(X[mask], y[mask])

        try:
            while True:
                train_idx: List[int] = []
                test_idx: List[int] = []

                for name, mask in masks:
                    indices = np.flatnonzero(mask)
                    train, test = next(iters[name])

                    train_idx.extend(indices[train])
                    test_idx.extend(indices[test])
                yield train_idx, test_idx
        except StopIteration:
            return
