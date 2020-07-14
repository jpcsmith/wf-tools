"""Tests for the DeepFingerprinting classifier."""
import pytest
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.compat.v1.keras.utils import plot_model
from lab.classifiers.varcnn import VarCNNClassifier
from lab.feature_extraction.trace import extract_sizes


@pytest.fixture(name="train_test_data_size")
def fixture_train_test_data_size(dataset) -> tuple:
    """Return a tuple of (x_train, x_test, y_train, y_test) in the
    closed-world setting.
    """
    traces, classes = dataset
    features = extract_sizes(traces)
    return train_test_split(features, classes, random_state=7141845)

def test_print_model():
    classifier = VarCNNClassifier(n_packet_features=3, n_classes=3)
    classifier.fit(
        np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        [0, 0, 1, 1, 2, 2])
    plot_model(classifier.model, to_file='final-model.png', show_shapes=True,
               show_layer_names=True)


# def test_df_on_sample(train_test_data):
#     """Simple sanity test."""
#     x_train, x_test, y_train, y_test = train_test_data
#
#     classifier = VarCNNClassifier(epochs=1)
#     classifier.fit(x_train, y_train)
#     # assert classifier.score(x_test, y_test) > 0.8
