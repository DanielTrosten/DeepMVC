from torch import nn

from models.base import BaseModel
from models.clustering_module import get_clustering_module
from lib.fusion import get_fusion_module
from register import register_model


@register_model
class SiMVC(BaseModel):
    def __init__(self, cfg):
        super(SiMVC, self).__init__(cfg)

        self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=self.fusion.output_size)

        self.encoder_outputs = self.fused = self.hidden = self.output = None

    @property
    def fusion_weights(self):
        w = getattr(self.fusion, "weights", None)
        if w is not None:
            w = nn.functional.softmax(w.squeeze(), dim=-1).detach()
        return w

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output
