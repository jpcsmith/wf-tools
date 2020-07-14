"""An implementation of the Var-CNN Classifiers from

    S. Bhat, D. Lu, A. Kwon, and S. Devadas, “Var-CNN: A Data-Efficient
    Website Fingerprinting Attack Based on Deep Learning,” Proceedings
    on Privacy Enhancing Technologies, vol. 2019, no. 4, pp. 292–310,
    2019, doi: https://doi.org/10.2478/popets-2019-0070.

This implementation uses code from the original paper, but modifies it
to be used more flexibly.

The original can be found at https://github.com/sanjit-bhat/Var-CNN.
"""
# pylint: disable=too-many-arguments,invalid-name,too-many-instance-attributes
import math
import time
import logging
from typing import FrozenSet, Optional

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from tensorflow.compat.v1 import keras, Variable
from tensorflow.compat.v1.keras import layers
import numpy as np


PARAMETERS = {'kernel_initializer': 'he_normal'}


class Crop(layers.Layer):
    """Crop the input along axis 1.
    """
    def __init__(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        **kwargs
    ):
        super().__init__(self, **kwargs)
        assert start is not None or end is not None
        self.start = start
        self.end = end
        # self.start = Variable(initial_value=start, trainable=False)
        # self.end = Variable(inital_value=end, trainable=False)

    def call(self, inputs):
        """Run the layer."""
        return inputs[:, self.start:self.end]

    def get_config(self) -> dict:
        """Return the configuration."""
        return {"start": self.start, "end": self.end}


# TODO: Allow saving the model for later
class VarCNNClassifier(BaseEstimator, ClassifierMixin):
    """Var-CNN classifier using a CNN with either timing or direction
    based features.
    """
    # TODO: We may need to re-evaluate how we combine the timing and direction
    # features in the ensemble, as we are no longer using simple directions.
    def __init__(
        self,
        base_patience: int = 5,
        mixture: FrozenSet[str] = frozenset(("dir", "metadata")),
        dilations: bool = True,
        epochs: int = 150,
        batch_size: int = 50,
        n_meta_features: int = 7,
        random_state=None,
    ):
        assert n_meta_features >= 0

        self.base_patience = base_patience
        self.mixture = mixture
        self.tag = "dir" if "dir" in mixture else "time"
        self.dilations = dilations
        self.model_name = "var-cnn"
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_meta_features = n_meta_features
        self.random_state = random_state
        self._logger = logging.getLogger(__name__ + "." + self.model_name)

    def _init_model(self, n_classes: int, n_features: int):
        """Initialise and return the model and callbacks."""
        use_metadata = self.n_meta_features > 0

        n_total_features = n_features + self.n_meta_features

        # Constructs dir or time ResNet
        block = dilated_basic_1d if self.dilations else basic_1d
        input_layer = keras.Input(
            shape=(n_total_features, ), name=(self.tag + '_input'))

        if use_metadata:
            reshape_layer = Crop(end=-self.n_meta_features)(input_layer)
        else:
            reshape_layer = input_layer
        reshape_layer = layers.Reshape((n_features, 1))(reshape_layer)
        output_layer = ResNet18(reshape_layer, self.tag, block=block)

        # Construct MLP for metadata
        if use_metadata:
            metadata_output = Crop(start=-self.n_meta_features)(input_layer)
            # metadata_input = keras.Input(
            #     shape=(self.n_meta_features, ), name='metadata_input')
            # consider this the embedding of all the metadata
            metadata_output = layers.Dense(32)(metadata_output)
            metadata_output = layers.BatchNormalization()(
                metadata_output)
            metadata_output = layers.Activation('relu')(metadata_output)

        # Forms input and output lists and possibly add final dense layer
        input_params = [input_layer]
        concat_params = [output_layer]
        combined = concat_params[0]

        if use_metadata:
            # input_params.append(metadata_input)
            concat_params.append(metadata_output)
            combined = layers.Concatenate()(concat_params)

        # Better to have final fc layer if combining multiple models
        if len(concat_params) > 1:
            combined = layers.Dense(1024)(combined)
            combined = layers.BatchNormalization()(combined)
            combined = layers.Activation('relu')(combined)
            combined = layers.Dropout(0.5)(combined)

        model_output = layers.Dense(units=n_classes, activation='softmax',
                                    name='model_output')(combined)

        model = keras.Model(inputs=input_params, outputs=model_output)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=keras.optimizers.Adam(0.001))

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc', factor=np.sqrt(0.1), cooldown=0, min_lr=1e-5,
                patience=self.base_patience, verbose=1),
            keras.callbacks.EarlyStopping(
                monitor='val_acc', patience=(2 * self.base_patience)),
            keras.callbacks.ModelCheckpoint(
                'model_weights.h5', monitor='val_acc', save_best_only=True,
                save_weights_only=True, verbose=1)
        ]

        return model, callbacks

    def _split_features(self, X) -> tuple:
        """Return a tuple of (X_traces, X_meta), with both feature sets
        reshaped as necessary.
        """
        n_trace_features = X.shape[1] - self.n_meta_features
        X_traces = X[:, :-self.n_meta_features].reshape(
            (-1, n_trace_features, 1))
        X_meta = X[:, -self.n_meta_features]
        return (X_traces, X_meta)

    def fit(self, X, y):
        """Fit the Var-CNN model.

        Parameters:
        ----------
        X : array-like of shape (n_samples, n_trace_features + n_meta_features)
            The training input samples.  For a positive n_meta_features,
            X is split and trace and meta features are passed to the
            appropriate CNN inputs.

        y : array-like of shape (n_samples, )
            The target class labels. Must be numeric.
        """
        X = check_array(X, accept_sparse=False,
                        ensure_min_features=(1 + self.n_meta_features))
        y = check_array(y, accept_sparse=False, ensure_2d=False)

        # Shuffle the features, as this is less error prone than
        # remembering to do it, and a fraction of the dataset is taken
        # for validation without shuffling.
        random_state = check_random_state(self.random_state)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
        X_traces, X_meta = self._split_features(X)

        n_classes = np.unique(y).size
        encoder = OneHotEncoder(sparse=False)
        # TODO: Is this reshape necessary?
        y = encoder.fit_transform(y.reshape(-1, 1))

        # pylint: disable=attribute-defined-outside-init
        self.n_features_ = X.shape[1]
        self.classes_ = encoder.categories_[0]
        self.model_, callbacks = self._init_model(
            n_classes, (self.n_features_ - self.n_meta_features))

        return self.model_

        # self._logger.info("Starting training ... ")
        # start_time = time.perf_counter()
        # self.model_.fit(
        #     {(self.tag + "_input"): X_traces, "metadata_input": X_meta},
        #     {"model_output": y}, validation_split=0.1, epochs=self.epochs,
        #     verbose=2, callbacks=callbacks, shuffle=False)
        # self._logger.info(
        #     "Training complete in %.2fs", (time.perf_counter() - start_time))
        # return self

    def predict(self, X):
        """Compute and save final predictions on test set."""
        check_is_fitted(self, ["model_", "n_features_"])
        X = check_array(
            X, accept_sparse=False, ensure_min_features=self.n_features_)
        X_traces, X_meta = self._split_features(X)

        test_size = X.shape[0]
        steps = math.ceil(test_size // self.batch_size)

        self._logger.info("Starting predictions ... ")
        start = time.perf_counter()

        predictions = self.model_.predict(
            [X_traces, X_meta], batch_size=self.batch_size, steps=steps,
            verbose=0)

        self._logger.info(
            "Prediction complete in %.2fs.", (time.perf_counter() - start))
        return predictions


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def dilated_basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
                     numerical_name=False, stride=None,
                     dilations=(1, 1)):
    """A one-dimensional basic residual block with dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead of
        chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the
        first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second
        conv layers
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if block > 0 and numerical_name:
        block_char = 'b{}'.format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def dilated_basic_1d_block(x):
        y = layers.Conv1D(
            filters, kernel_size, padding='causal', strides=stride,
            dilation_rate=dilations[0], use_bias=False,
            name='res{}{}_branch2a_{}'.format(stage_char, block_char, suffix),
            **PARAMETERS)(x)
        y = layers.BatchNormalization(
            epsilon=1e-5, name='bn{}{}_branch2a_{}'.format(
                stage_char, block_char, suffix))(y)
        y = layers.Activation('relu', name='res{}{}_branch2a_relu_{}'.format(
            stage_char, block_char, suffix))(y)

        y = layers.Conv1D(
            filters, kernel_size, padding='causal', use_bias=False,
            dilation_rate=dilations[1], name='res{}{}_branch2b_{}'.format(
                stage_char, block_char, suffix),
            **PARAMETERS)(y)
        y = layers.BatchNormalization(
            epsilon=1e-5, name='bn{}{}_branch2b_{}'.format(
                stage_char, block_char, suffix))(y)

        if block == 0:
            shortcut = layers.Conv1D(
                filters, 1, strides=stride, use_bias=False,
                name='res{}{}_branch1_{}'.format(
                    stage_char, block_char, suffix), **PARAMETERS)(x)
            shortcut = layers.BatchNormalization(
                epsilon=1e-5, name='bn{}{}_branch1_{}'.format(
                    stage_char, block_char,
                    suffix))(shortcut)
        else:
            shortcut = x

        y = layers.Add(
            name='res{}{}_{}'.format(stage_char, block_char, suffix))(
                [y, shortcut])
        y = layers.Activation(
            'relu', name='res{}{}_relu_{}'.format(
                stage_char, block_char, suffix))(y)
        return y

    return dilated_basic_1d_block


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def basic_1d(filters, suffix, stage=0, block=0, kernel_size=3,
             numerical_name=False, stride=None, dilations=(1, 1)):
    """A one-dimensional basic residual block without dilations.

    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param numerical_name: if true, uses numbers to represent blocks instead
        of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the
        first conv layer, default derives stride from block id
    :param dilations: tuple representing amount to dilate first and second
        conv layers
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    dilations = (1, 1)

    if block > 0 and numerical_name:
        block_char = 'b{}'.format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    def basic_1d_block(x):
        y = layers.Conv1D(
            filters, kernel_size, padding='same', strides=stride,
            dilation_rate=dilations[0], use_bias=False,
            name='res{}{}_branch2a_{}'.format(stage_char, block_char,
                                              suffix), **PARAMETERS)(x)
        y = layers.BatchNormalization(
            epsilon=1e-5, name='bn{}{}_branch2a_{}'.format(
                stage_char, block_char, suffix))(y)
        y = layers.Activation('relu', name='res{}{}_branch2a_relu_{}'.format(
            stage_char, block_char, suffix))(y)

        y = layers.Conv1D(
            filters, kernel_size, padding='same', use_bias=False,
            dilation_rate=dilations[1],
            name='res{}{}_branch2b_{}'.format(
                stage_char, block_char, suffix), **PARAMETERS)(y)
        y = layers.BatchNormalization(
            epsilon=1e-5, name='bn{}{}_branch2b_{}'.format(
                stage_char, block_char, suffix))(y)

        if block == 0:
            shortcut = layers.Conv1D(
                filters, 1, strides=stride, use_bias=False,
                name='res{}{}_branch1_{}'.format(
                    stage_char, block_char, suffix),
                **PARAMETERS)(x)
            shortcut = layers.BatchNormalization(
                epsilon=1e-5, name='bn{}{}_branch1_{}'.format(
                    stage_char, block_char, suffix))(shortcut)
        else:
            shortcut = x

        y = layers.Add(name='res{}{}_{}'.format(
            stage_char, block_char, suffix))([y, shortcut])
        y = layers.Activation('relu', name='res{}{}_relu_{}'.format(
            stage_char, block_char, suffix))(y)

        return y

    return basic_1d_block


# Code for standard ResNet model is based on
# https://github.com/broadinstitute/keras-resnet
def ResNet18(inputs, suffix, blocks=None, block=None, numerical_names=None):
    """Constructs a `keras.models.Model` object using the given block count.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of
        `keras_resnet.blocks.basic_2d`)
    :param numerical_names: list of bool, same size as blocks, used to
        indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`)
        or classification output (if `include_top=True`)
    """
    if blocks is None:
        blocks = [2, 2, 2, 2]
    if block is None:
        block = dilated_basic_1d
    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    x = layers.ZeroPadding1D(padding=3, name='padding_conv1_' + suffix)(inputs)
    x = layers.Conv1D(
        64, 7, strides=2, use_bias=False, name='conv1_' + suffix)(x)
    x = layers.BatchNormalization(
        epsilon=1e-5, name='bn_conv1_' + suffix)(x)
    x = layers.Activation('relu', name='conv1_relu_' + suffix)(x)
    x = layers.MaxPooling1D(
        3, strides=2, padding='same', name='pool1_' + suffix)(x)

    features = 64
    outputs = []

    for stage_id, iterations in enumerate(blocks):
        x = block(features, suffix, stage_id, 0, dilations=(1, 2),
                  numerical_name=False)(x)
        for block_id in range(1, iterations):
            x = block(features, suffix, stage_id, block_id, dilations=(4, 8),
                      numerical_name=(
                              block_id > 0 and numerical_names[stage_id]))(
                x)

        features *= 2
        outputs.append(x)

    x = layers.GlobalAveragePooling1D(name='pool5_' + suffix)(x)
    return x
