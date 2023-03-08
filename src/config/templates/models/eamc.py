from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion, optimizer, encoder, layers, clustering_module
from config.templates.models import BaseModel, DDCLoss


class EAMCLoss(DDCLoss):
    weights: Optional[List[float]] = [1, 1, 1, 1, 10, 1]
    funcs: str = "DDC1|DDC2Flipped|DDC3|EAMCAttention|EAMCGenerator|EAMCDiscriminator"


class Attention(Config):
    tau: float = 10.0
    mlp_config: encoder.Encoder = encoder.Encoder(
        layers=[
            layers.Dense(n_units=100),
            layers.Dense(n_units=50),
        ]
    )

    # Placeholder
    n_views: int = None


class Discriminator(Config):
    mlp_config: encoder.Encoder = encoder.Encoder(
        layers=[
            layers.Dense(n_units=256, activation="leaky_relu", activation_params=dict(neg_slope=0.2)),
            layers.Dense(n_units=256, activation="leaky_relu", activation_params=dict(neg_slope=0.2)),
            layers.Dense(n_units=128, activation="leaky_relu", activation_params=dict(neg_slope=0.2)),
        ],
    )


class EAMC(BaseModel):
    attention_config: Optional[Attention] = Attention()
    fusion_config: Optional[fusion.Fusion] = None
    discriminator_config: Discriminator = Discriminator()

    cm_config: clustering_module.ClusteringModule = clustering_module.DDC()
    loss_config: EAMCLoss = EAMCLoss()

    encoder_kernel_width_config: kernel_width.KernelWidth = kernel_width.MedianDistance()
    fused_kernel_width_config: kernel_width.KernelWidth = kernel_width.MedianDistance()

    # Optimizers for (generator + clustering module) and discriminator
    # optimizer_config: List[optimizer.Optimizer] = [
    #     optimizer.Optimizer(),
    #     optimizer.Optimizer(),
    # ]
    optimizer_config: Literal[None] = None
