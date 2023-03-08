from config.config import Config
from config.templates.encoder import Encoder


class Fusion(Config):
    n_views: int = None  # Placeholder


class WeightedMean(Fusion):
    trainable_weights: bool = True


class Concat(Fusion):
    pass


class MLPFusion(Fusion):
    mlp_config: Encoder = Encoder(
        layers="projection_head",
        input_size=None,
    )
