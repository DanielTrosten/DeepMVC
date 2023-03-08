from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion, optimizer, encoder, layers, clustering_module
from config.templates.models import BaseModel, BaseLoss


class DMSCLoss(BaseLoss):
    funcs: str = "DMSC1|DMSC2|DMSC3"
    weights: Optional[List[float]] = [1.0, 1.0, 1.0]


class DMSCPreTrainLoss(BaseLoss):
    funcs: str = "DMSC3"


class DMSC(BaseModel):
    n_samples: int
    decoder_configs: List[encoder.Encoder]
    loss_config: DMSCLoss = DMSCLoss()
    pre_train_loss_config: Optional[DMSCPreTrainLoss] = DMSCPreTrainLoss()

