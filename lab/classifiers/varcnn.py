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
# pylint: disable=too-few-public-methods
import logging
from typing import Optional

import numpy as np
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras.wrappers.scikit_learn import KerasClassifier


PARAMETERS = {'kernel_initializer': 'he_normal'}


class Crop(layers.Layer):
    """Crop the input along axis 1, returning inputs[:, start:end].
    One of start and end can be none, in which case it's equivalent to
    inputs[:, :end] and inputs[:, start:] respectively.
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

    def call(self, inputs):
        """Run the layer."""
        return inputs[:, self.start:self.end]

    def get_config(self) -> dict:
        """Return the configuration."""
        return {"start": self.start, "end": self.end}


def build_model(
    n_classes: int,
    n_packet_features: int,
    n_meta_features: int = 7,
    dilations: bool = True,
    tag: str = "varcnn",
):
    """Build the Var-CNN model.

    The resulting model takes a single input of shape
    (n_samples, n_packet_features + n_meta_features). The meta features
    must be the rightmost (last) features in the matrix.  The model
    handles separating the two types of features and reshaping them
    as necessary.

    Parameters:
    -----------
    n_classes :
        The number of classes to be predicted.

    n_packet_features :
        The number of packet features such as the number of interarrival
        times or the number of packet directions or sizes.

    n_meta_features:
        The number of meta features such as total packet counts, total
        transmission duration, etc.
    """
    use_metadata = n_meta_features > 0

    # Constructs dir or time ResNet
    input_layer = keras.Input(
        shape=(n_packet_features + n_meta_features, ), name="input")

    layer = (Crop(end=n_packet_features)(input_layer)
             if use_metadata else input_layer)
    layer = layers.Reshape((n_packet_features, 1))(layer)
    output_layer = ResNet18(
        layer, tag, block=(dilated_basic_1d if dilations else basic_1d))

    concat_params = [output_layer]
    combined = concat_params[0]

    # Construct MLP for metadata
    if use_metadata:
        metadata_output = Crop(start=-n_meta_features)(input_layer)
        # consider this the embedding of all the metadata
        metadata_output = layers.Dense(32)(metadata_output)
        metadata_output = layers.BatchNormalization()(
            metadata_output)
        metadata_output = layers.Activation('relu')(metadata_output)

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

    model = keras.Model(inputs=input_layer, outputs=model_output)
    model.compile(
        loss='categorical_crossentropy', metrics=['accuracy'],
        optimizer=keras.optimizers.Adam(0.001))

    return model


# TODO: Allow saving the model for later
class VarCNNClassifier(KerasClassifier):
    """Var-CNN classifier using a CNN with either timing or direction
    based features.

    See `varcnn.build_model` for other arguments.
    """
    def __init__(self, base_patience: int = 5, **kwargs):
        super().__init__(
            build_fn=build_model,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy', factor=np.sqrt(0.1), cooldown=0,
                    min_lr=1e-5, patience=base_patience, verbose=1),
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=(2 * base_patience)),
            ],
            **kwargs)

    # Code taken from KerasClassifier and Sequential, as Models do not support
    # predict_proba
    def predict_proba(self, x, **kwargs):
        """Returns class probability estimates for the given test data.
        """
        kwargs = self.filter_sk_params(keras.Model.predict, kwargs)
        probs = self.model.predict(x, **kwargs)

        if probs.min() < 0. or probs.max() > 1.:
            logging.warning('Network returning invalid probability values. '
                            'The last layer might not normalize predictions '
                            'into probabilities '
                            '(like softmax or sigmoid would).')

        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs


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
