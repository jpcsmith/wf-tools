"""An implementation of the DF classifier from

    P. Sirinam, M. Imani, M. Juarez, and M. Wright, "Deep Fingerprinting:
    Undermining Website Fingerprinting Defenses with Deep Learning," in
    Proceedings of the 2018 ACM SIGSAC Conference on Computer and
    Communications Security, Toronto, Canada, 2018, pp. 1928–1943,
    doi: 10.1145/3243734.3243768.

The implementation uses code from the original paper, specifically the
code from 'Model_NoDef.py'.  The original can be found at
https://github.com/deep-fingerprinting/df
"""
# pylint: disable=too-many-statements,too-few-public-methods
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers, initializers

from lab.classifiers.wrappers import ModifiedKerasClassifier


def build_model(
    n_features: int, n_classes: int, metric="accuracy",
    learning_rate: float = 0.002
):
    """Create and return the DeepFingerprinting Model."""
    model = keras.Sequential()
    # Block1
    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model.add(layers.Reshape((n_features, 1), input_shape=(n_features,)))
    model.add(layers.Conv1D(
        filters=filter_num[1], kernel_size=kernel_size[1],
        strides=conv_stride_size[1], padding='same', name='block1_conv1'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ELU(alpha=1.0, name='block1_adv_act1'))
    model.add(layers.Conv1D(
        filters=filter_num[1], kernel_size=kernel_size[1],
        strides=conv_stride_size[1], padding='same', name='block1_conv2'))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.ELU(alpha=1.0, name='block1_adv_act2'))
    model.add(layers.MaxPooling1D(
        pool_size=pool_size[1], strides=pool_stride_size[1], padding='same',
        name='block1_pool'))
    model.add(layers.Dropout(0.1, name='block1_dropout'))

    model.add(layers.Conv1D(
        filters=filter_num[2], kernel_size=kernel_size[2],
        strides=conv_stride_size[2], padding='same', name='block2_conv1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block2_act1'))

    model.add(layers.Conv1D(
        filters=filter_num[2], kernel_size=kernel_size[2],
        strides=conv_stride_size[2], padding='same', name='block2_conv2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block2_act2'))
    model.add(layers.MaxPooling1D(
        pool_size=pool_size[2], strides=pool_stride_size[3], padding='same',
        name='block2_pool'))
    model.add(layers.Dropout(0.1, name='block2_dropout'))

    model.add(layers.Conv1D(
        filters=filter_num[3], kernel_size=kernel_size[3],
        strides=conv_stride_size[3], padding='same', name='block3_conv1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block3_act1'))
    model.add(layers.Conv1D(
        filters=filter_num[3], kernel_size=kernel_size[3],
        strides=conv_stride_size[3], padding='same', name='block3_conv2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block3_act2'))
    model.add(layers.MaxPooling1D(
        pool_size=pool_size[3], strides=pool_stride_size[3], padding='same',
        name='block3_pool'))
    model.add(layers.Dropout(0.1, name='block3_dropout'))

    model.add(layers.Conv1D(
        filters=filter_num[4], kernel_size=kernel_size[4],
        strides=conv_stride_size[4], padding='same', name='block4_conv1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block4_act1'))
    model.add(layers.Conv1D(
        filters=filter_num[4], kernel_size=kernel_size[4],
        strides=conv_stride_size[4], padding='same', name='block4_conv2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='block4_act2'))
    model.add(layers.MaxPooling1D(
        pool_size=pool_size[4], strides=pool_stride_size[4], padding='same',
        name='block4_pool'))
    model.add(layers.Dropout(0.1, name='block4_dropout'))

    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dense(
        512, kernel_initializer=initializers.glorot_uniform(seed=0),
        name='fc1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='fc1_act'))

    model.add(layers.Dropout(0.7, name='fc1_dropout'))

    model.add(layers.Dense(
        512, kernel_initializer=initializers.glorot_uniform(seed=0),
        name='fc2'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu', name='fc2_act'))

    model.add(layers.Dropout(0.5, name='fc2_dropout'))

    model.add(layers.Dense(
        n_classes, kernel_initializer=initializers.glorot_uniform(seed=0),
        name='fc3'))
    model.add(layers.Activation('softmax', name="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adamax(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-08, decay=0.0
        ),
        metrics=[metric])

    return model


class DeepFingerprintingClassifier(ModifiedKerasClassifier):
    """Website fingerprinting classifer using a CNN."""
    def __init__(self, **kwargs):
        if "build_fn" in kwargs:
            del kwargs["build_fn"]
        super().__init__(build_fn=build_model, **kwargs)

    def __repr__(self) -> str:
        params = self.filter_sk_params(build_model)
        return "DeepFingerprintingClassifier({})".format(
            ", ".join(f"{arg}={value!r}" for arg, value in params.items()))
