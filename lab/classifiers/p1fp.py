"""An implementation of the p1-FP classifiers from:

    S. E. Oh, S. Sunkam, and N. Hopper, "p1-FP: Extraction,
    Classification, and Prediction of Website Fingerprints with Deep
    Learning," Proceedings on Privacy Enhancing Technologies, vol. 2019,
    no. 3, pp. 191–209, Jul. 2019, doi: 10.2478/popets-2019-0043.

This implementation uses the code from the original paper, but modifies
them to be reusable.

The original can be found at https://github.com/seeunoh2/pFP
"""
import numpy as np
import tensorflow as tf
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
try:
    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.normalization import local_response_normalization
except ImportError:
    USE_TFLEARN = False
else:
    USE_TFLEARN = True

from tensorflow.compat.v1 import keras, nn
from tensorflow.compat.v1.keras import layers

from lab.classifiers.wrappers import ModifiedKerasClassifier


def build_cnn_model(n_features: int, n_classes: int):
    """Build the P1FP(C) model using Keras."""
    model = keras.Sequential()
    model.add(layers.Reshape((1, n_features, 1), input_shape=(n_features, ),
              name="input"))

    model.add(layers.Conv2D(
        128, 12, activation="relu", kernel_regularizer="l2", padding="same"))
    model.add(layers.MaxPool2D(10, padding="same"))
    model.add(layers.Lambda(nn.local_response_normalization))

    model.add(layers.Conv2D(
        128, 12, activation="relu", kernel_regularizer="l2", padding="same"))
    model.add(layers.MaxPool2D(10, padding="same"))
    model.add(layers.Lambda(nn.local_response_normalization))

    # It is flattened for the computation regardless, however tflearn retained
    # the flattened result whereas keras does not
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="tanh"))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(n_classes, activation="softmax", name="target"))

    learning_rate = keras.optimizers.schedules.ExponentialDecay(
        0.05, decay_steps=1000, decay_rate=0.96)
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.TopKCategoricalAccuracy(3), "accuracy"])

    return model


class KerasP1FPClassifierC(ModifiedKerasClassifier):
    """A wrapper around the Keras implementation of the classifier.

    See p1fp.build_cnn_model for parameter details.
    """
    def __init__(self, **kwargs):
        super().__init__(build_fn=build_cnn_model, **kwargs)


if USE_TFLEARN:
    class P1FPClassifierC(BaseEstimator, ClassifierMixin):
        """p1-FP(C) classifier using a convolution neural network.
        """
        def __init__(self, n_epoch: int = 40, snapshot_step: int = 100):
            self.n_epoch = n_epoch
            self.snapshot_step = snapshot_step

        def fit(self, X, y, name: str = "p1-fp(C)") -> "P1FPClassifierC":
            """Fit the convolution neural network with the given data."""
            X = check_array(X, accept_sparse=False)
            y = check_array(y, accept_sparse=False, ensure_2d=False)

            n_features = X.shape[1]
            n_classes = np.unique(y).size

            X = X.reshape([-1, 1, n_features, 1])

            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(y.reshape(-1, 1))

            tf.compat.v1.reset_default_graph()
            network = input_data(shape=[None, 1, n_features, 1], name='input')
            network = conv_2d(network, 128, 12, activation='relu',
                              regularizer="L2")
            network = max_pool_2d(network, 10)
            network = local_response_normalization(network)

            network = conv_2d(network, 128, 12, activation='relu',
                              regularizer="L2")
            network = max_pool_2d(network, 10)
            network = local_response_normalization(network)

            network = fully_connected(network, 256, activation='tanh')
            network = dropout(network, 0.8)

            softmax = fully_connected(network, n_classes, activation='softmax')

            sgd = tflearn.SGD(learning_rate=0.05, lr_decay=0.96,
                              decay_step=1000)
            top_k = tflearn.metrics.Top_k(3)
            network = tflearn.regression(
                softmax, optimizer=sgd, loss='categorical_crossentropy',
                name='target', metric=top_k)

            # pylint: disable=attribute-defined-outside-init
            self.n_features_ = n_features
            self.classes_ = encoder.categories_[0]
            self.model_ = tflearn.DNN(network, tensorboard_verbose=0)

            self.model_.fit({'input': X}, {'target': y}, validation_set=0.1,
                            n_epoch=self.n_epoch,
                            snapshot_step=self.snapshot_step, run_id=name)
            return self

        def save_model(self, path: str) -> "P1FPClassifierC":
            """Save the model to the specified path."""
            check_is_fitted(self, ["model_"])
            self.model_.save(path)
            return self

        def predict_proba(self, X):
            """Make probability predictions for the provided data."""
            check_is_fitted(self, ["model_", "n_features_"])

            X = check_array(
                X, accept_sparse=False, ensure_min_features=self.n_features_)
            X = X.reshape([-1, 1, self.n_features_, 1])
            return self.model_.predict(X)

        def predict(self, X):
            """Predict the labels for provided test data X."""
            check_is_fitted(self, ["classes_"])
            result = self.predict_label(X)
            return np.array([self.classes_[ranking[0]] for ranking in result])

        def predict_label(self, X):
            """Returns the rankings of the labels (indexes in classes_)
            based on their probabilities.
            """
            check_is_fitted(self, ["model_", "n_features_"])

            X = check_array(
                X, accept_sparse=False, ensure_min_features=self.n_features_)
            X = X.reshape([-1, 1, self.n_features_, 1])
            return self.model_.predict_label(X)

    def onehot(label_array, n_classes):
        """Perform a one-hot encoding. Taken from the original repo."""
        converted_arr = np.zeros(shape=(len(label_array), n_classes))
        i = 0
        neg = 0
        for label in label_array:
            if int(label) == -1:
                neg += 1
            converted_arr[i][int(label)] = 1
            i += 1
        count = 0

        for item in converted_arr:
            if item[-1] == 1:
                count += 1
        return converted_arr
