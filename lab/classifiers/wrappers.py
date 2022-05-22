"""Wrappers for working with classifiers."""
from tensorflow.compat.v1.keras.wrappers import scikit_learn


class ModifiedKerasClassifier(scikit_learn.KerasClassifier):
    """A wrapper around KerasClassifier that handles the validation set."""
