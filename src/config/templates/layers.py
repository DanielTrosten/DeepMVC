from typing import Union, List
from typing_extensions import Literal

from config.config import Config


class Layer(Config):
    activation: str = None
    activation_params: dict = None


class Dense(Layer):
    n_units: int
    bias: bool = True


class Conv(Layer):
    out_channels: int
    kernel_size: Union[List[int], int]
    stride: Union[List[int], int] = 1
    padding: str = "valid"


class ConvTranspose(Layer):
    out_channels: int
    kernel_size: Union[List[int], int]
    padding: str = "valid"


class MaxPool(Layer):
    kernel_size: Union[List[int], int]
    padding: str = "valid"


class UpSample(Layer):
    kernel_size: int
    mode: str = "nearest"


class AdaptiveAvgPool(Layer):
    output_size: List[int] = [1, 1]


class BatchNormalization(Layer):
    affine: bool = True


class Activation(Layer):
    pass


class Flatten(Layer):
    pass


RELU = Activation(activation="relu")
SOFTMAX = Activation(activation="softmax")
SIGMOID = Activation(activation="sigmoid")


DEFAULT_LAYERS = {
    "cnn_tiny": [
        Conv(out_channels=32, kernel_size=3),
        BatchNormalization(),
        RELU,
        Conv(out_channels=32, kernel_size=3),
        BatchNormalization(),
        RELU,
        Conv(out_channels=32, kernel_size=3),
        BatchNormalization(),
    ],

    "cnn_tiny_decoder": [
        ConvTranspose(out_channels=32, kernel_size=3),
        BatchNormalization(),
        RELU,

        ConvTranspose(out_channels=32, kernel_size=3),
        BatchNormalization(),
        RELU,

        ConvTranspose(out_channels=1, kernel_size=3, activation="sigmoid"),
    ],

    "cnn_small": [
        Conv(out_channels=64, kernel_size=3, activation="relu"),
        Conv(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,
        MaxPool(kernel_size=2),

        Conv(out_channels=64, kernel_size=3, activation="relu"),
        Conv(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,
        MaxPool(kernel_size=2),
    ],

    "cnn_small_decoder": [
        UpSample(kernel_size=2),

        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,

        UpSample(kernel_size=2),

        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=1, kernel_size=3, activation="sigmoid"),
    ],

    "cnn_large": [
        Conv(out_channels=64, kernel_size=3, activation="relu"),
        Conv(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,
        MaxPool(kernel_size=2),

        Conv(out_channels=64, kernel_size=3, activation="relu"),
        Conv(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,
        MaxPool(kernel_size=2),

        Conv(out_channels=64, kernel_size=3, activation="relu"),
        Conv(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,
        MaxPool(kernel_size=2),
    ],

    "cnn_large_decoder": [
        UpSample(kernel_size=2),
        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,

        UpSample(kernel_size=2),
        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,

        UpSample(kernel_size=2),
        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=64, kernel_size=3, activation=None),
        BatchNormalization(),
        RELU,

        ConvTranspose(out_channels=64, kernel_size=3, activation="relu"),
        ConvTranspose(out_channels=1, kernel_size=3, activation="sigmoid"),
    ],

    "dense_5": [
        Dense(n_units=1024),
        BatchNormalization(),
        RELU,

        Dense(n_units=1024),
        BatchNormalization(),
        RELU,

        Dense(n_units=1024),
        BatchNormalization(),
        RELU,

        Dense(n_units=1024),
        BatchNormalization(),
        RELU,

        Dense(n_units=256),
    ],

    "dense_2d": [
        Dense(n_units=8),
        BatchNormalization(),
        RELU,

        Dense(n_units=8, activation="relu"),
    ],

    "dense_2d_decoder": [
        Dense(n_units=8),
        BatchNormalization(),
        RELU,

        Dense(n_units=2)

    ],

    "projection_head": [
        Dense(n_units=-1),
        BatchNormalization(),
        RELU,
        Dense(n_units=-1, activation=None, bias=False),
    ],

    "linear_projection": [
        Dense(n_units=-1, activation=None, bias=False)
    ],
}

