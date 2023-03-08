from typing import List, Optional

from config.templates.clustering_module import DDC
from config.templates.optimizer import Optimizer
from config.templates.models import BaseLoss, BaseModel


class MIMVCLoss(BaseLoss):
    funcs: str = "DDC1|DDC2|DDC3|MIContrast"
    alpha: float = 9.0
    weights: Optional[List[float]] = [1.0, 1.0, 1.0, 0.1]


class MIMVC(BaseModel):
    cm_config: DDC = DDC()
    loss_config: MIMVCLoss = MIMVCLoss()

    pre_train_loss_config: MIMVCLoss = MIMVCLoss(funcs="MIContrast", weights=None)
    pre_train_optimizer_config: Optimizer = Optimizer()
