from config.config import Config

from typing_extensions import Literal


class KernelWidth(Config):
    initial_value: float = 1.0


class Constant(KernelWidth):
    pass


class AMISE(KernelWidth):
    initial_value: None = None
    std_estimator: Literal["global", "within_cluster"] = "within_cluster"

    # Placeholder
    batch_size: int = None


class MomentumAMISE(KernelWidth):
    momentum: float = 0.9
    std_estimator: Literal["global", "within_cluster"] = "within_cluster"

    # Placeholder
    batch_size: int = None


class MedianDistance(KernelWidth):
    initial_value: None = None
    rel_sigma: float = 0.15


class MomentumMedianDistance(KernelWidth):
    rel_sigma: float = 0.15
    momentum: float = 0.9
