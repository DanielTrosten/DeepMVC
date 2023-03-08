from typing import List, Optional

from config.config import Config
from config.templates import fusion, encoder, optimizer


class BaseLoss(Config):
    # Terms to use in the loss, separated by '|'. E.g. "DDC1|DDC2|DDC3" for the DDC clustering loss
    funcs: str
    # Optional weights for the loss terms. Set to None to have all weights equal to 1.
    weights: Optional[List[float]] = None

    # Placeholders
    n_views: int = None
    batch_size: int = None
    n_clusters: int = None


class DDCLoss(BaseLoss):
    funcs: str = "DDC1|DDC2|DDC3"
    numerator_epsilon: bool = True


class BaseModel(Config):
    # Encoder network config
    encoder_configs: List[encoder.Encoder]
    fusion_config: fusion.Fusion = fusion.WeightedMean()
    # Loss function config
    loss_config: BaseLoss
    # Optimizer config
    optimizer_config: optimizer.Optimizer = optimizer.Optimizer()

    # Pre-train stuff
    pre_train_loss_config: BaseLoss = None
    pre_train_optimizer_config: optimizer.Optimizer = None

    # Initial weights identifier
    initial_weights: str = None

    # Placeholders
    batch_size: int = None
    n_views: int = None
    n_clusters: int = None



