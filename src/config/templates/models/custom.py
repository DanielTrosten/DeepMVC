from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion, optimizer, encoder, layers, clustering_module
from config.templates.models import BaseModel, BaseLoss, DDCLoss


# ======================================================================================================================
# Losses
# ======================================================================================================================

class CAELoss(DDCLoss):
    funcs: str = "DDC1|DDC2|DDC3|Contrast|MSE"
    weights: Optional[List[float]] = [1.0, 1.0, 1.0, 0.1, 1.0]
    tau: float = 0.1


class CAEKMLoss(BaseLoss):
    funcs: str = "Contrast|MSE"
    weights: Optional[List[float]] = [0.1, 1.0]
    tau: float = 0.1


# ======================================================================================================================
# Models
# ======================================================================================================================

class CAE(BaseModel):
    decoder_configs: List[encoder.Encoder]
    projector_config: Optional[encoder.Encoder] = None
    fusion_config: fusion.Fusion = fusion.WeightedMean()
    cm_config: clustering_module.ClusteringModule = clustering_module.DDC()

    loss_config: CAELoss = CAELoss()
    pre_train_loss_config: Union[CAELoss, CAEKMLoss] = CAEKMLoss()


class CAEKM(BaseModel):
    decoder_configs: List[encoder.Encoder]
    projector_config: Optional[encoder.Encoder] = None
    loss_config: CAEKMLoss = CAEKMLoss()
    latent_norm: Optional[Literal["softmax", "l2"]] = None
