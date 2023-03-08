from typing import List, Union, Optional, Callable
from typing_extensions import Literal

from config import Config
from config.templates.layers import Layer


class Encoder(Config):
    input_size: Optional[List[int]]
    layers: Union[str, Callable, List[Layer]]
    output_normalization: Literal["l2", "softmax"] = None


class CNN(Encoder):
    # Network layers
    layers: str = "cnn_small"


class MLP(Encoder):
    # Units in the network layers
    layers: str = "dense_small"
