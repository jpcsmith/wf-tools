"""Wrappers for working with classifiers."""
import numpy as np
from tensorflow.compat.v1.keras.wrappers import scikit_learn
from tensorflow.python.keras.utils.np_utils import to_categorical


class ModifiedKerasClassifier(scikit_learn.KerasClassifier):
    """A wrapper around KerasClassifier that handles the validation set."""
    def fit(self, x, y, **kwargs):
        """Fit the model, reshaping validation_data if present."""
        # Reshape the validation set args in a similar fashion to how y will be
        # reshaped. Only necessary because varcnn uses categorical cross entropy
        if "validation_data" in kwargs:
            val_x, val_y = kwargs["validation_data"]
            if len(val_y.shape) == 1:
                # Encode the the validation y using the classes from the
                # training set
                classes_ = np.unique(y)
                val_y = np.searchsorted(classes_, val_y)
                # Make it categorical
                val_y = to_categorical(val_y)

                kwargs["validation_data"] = (val_x, val_y)

        super().fit(x, y, **kwargs)
