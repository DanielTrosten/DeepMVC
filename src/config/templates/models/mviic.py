from typing import List, Optional

from config.templates.models import BaseLoss, BaseModel
from config.templates import fusion, encoder, layers


class MvIICLoss(BaseLoss):
    funcs: str = "IICClustering|IICOverClustering"
    weights: Optional[List[float]] = [1.0, 0.01]
    lam: float = 1.5


class MvIIC(BaseModel):
    clustering_head_config: encoder.Encoder
    overclustering_head_config: Optional[encoder.Encoder]
    n_overclustering_heads: int = 5
    loss_config: MvIICLoss = MvIICLoss()
