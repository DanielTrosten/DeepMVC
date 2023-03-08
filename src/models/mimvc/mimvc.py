from torch import nn

from models.base import BaseModelPreTrain
from register import register_model
from lib.fusion import get_fusion_module
from models.clustering_module import get_clustering_module


@register_model
class MIMVC(BaseModelPreTrain):
    def __init__(self, cfg):
        super(MIMVC, self).__init__(cfg)

        self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=self.fusion.output_size)

        self.encoder_outputs = None
        self.fused = None
        self.hidden = None
        self.output = None

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.encoder_outputs = [nn.functional.softmax(x, dim=1) for x in self.encoder_outputs]

        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output
