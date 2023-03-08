from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion, optimizer, encoder, layers, clustering_module
from config.templates.models import BaseModel, BaseLoss


# ======================================================================================================================
# MvSCN
# ======================================================================================================================

class MVSCNLoss(BaseLoss):
    funcs: str = "MVSCN1|MVSCN2"
    weights: Optional[List[float]] = [0.999, 0.001]


class MvSCN(BaseModel):
    siam_dir: str
    siam_ckpt: str = "best.ckpt"

    aff_sigma: Optional[float] = None
    aff_n_neighbors: int = 5
    aff_n_scale_neighbors: Optional[int] = 2

    head_configs: Union[encoder.Encoder, List[encoder.Encoder]] = encoder.Encoder(layers=[layers.Dense(n_units=-1)])

    fusion_config: None = None
    loss_config: MVSCNLoss = MVSCNLoss()


# ======================================================================================================================
# Siamese Network
# ======================================================================================================================

class SiamLoss(BaseLoss):
    funcs: str = "Siamese"


class SiameseNet(BaseModel):
    loss_config: SiamLoss = SiamLoss()
    fusion_config: Literal[None] = None
    head_configs: Union[encoder.Encoder, List[encoder.Encoder]] = encoder.Encoder(layers=[layers.Dense(n_units=-1)])
