from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion, optimizer, encoder, layers, clustering_module
from config.templates.models import BaseModel, BaseLoss


class MultiVAELoss(BaseLoss):
    funcs: str = "MVAECont|MVAEDisc|MVAERec"
    cont_max_capacity: float = 5.0
    iters_add_capacity: int = 25000
    weights: Optional[List[float]] = [30.0, 30.0, 1.0]


class MultiVAE(BaseModel):
    temperature: float = 0.67
    cont_dim: int = 10
    hidden_dim: int = 256
    decoder_configs: List[encoder.Encoder]

    loss_config: MultiVAELoss = MultiVAELoss()
    optimizer_config: optimizer.Optimizer = optimizer.Optimizer(learning_rate=5e-4)
