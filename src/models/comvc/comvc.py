from torch import nn

from models.simvc.simvc import SiMVC
from lib import encoder
from register import register_model


@register_model
class CoMVC(SiMVC):
    def __init__(self, cfg):
        super(CoMVC, self).__init__(cfg)

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = encoder.Encoder(cfg.projector_config)

        self.projections = None

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.projections = [self.projector(x) for x in self.encoder_outputs]
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output
