from typing import List, Union, Optional
from typing_extensions import Literal

from config.templates import fusion, encoder, clustering_module
from config.templates.models import BaseModel, DDCLoss


class CoMVCLoss(DDCLoss):
    funcs: str = "DDC1|DDC2|DDC3|Contrast"
    weights: Optional[List[float]] = [1.0, 1.0, 1.0, 0.1]
    tau: float = 0.1
    contrast_adaptive_weight: bool = True


class SiMVC(BaseModel):
    fusion_config: fusion.Fusion = fusion.WeightedMean()
    cm_config: clustering_module.ClusteringModule = clustering_module.DDC()
    loss_config: DDCLoss = DDCLoss()


class CoMVC(SiMVC):
    projector_config: Optional[encoder.Encoder] = None
    loss_config: CoMVCLoss = CoMVCLoss()
