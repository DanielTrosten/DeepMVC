import torch as th
from torch import nn

from register import register_model
from models.base import BaseModelKMeans
from lib.encoder import Encoder


@register_model
class MvIIC(BaseModelKMeans):
    def __init__(self, cfg):
        super(MvIIC, self).__init__(cfg)

        enc_sizes = self.encoders.output_sizes
        assert all([enc_sizes[0] == s for s in enc_sizes]), "MvIIC expects all encoders to have same output size."

        self.clustering_head = Encoder(cfg.clustering_head_config, input_size=enc_sizes[0])

        if (cfg.overclustering_head_config is not None) and (cfg.n_overclustering_heads > 0):
            self.overclustering_heads = nn.ModuleList([
                Encoder(cfg.overclustering_head_config, input_size=enc_sizes[0]) for _ in range(cfg.n_overclustering_heads)
            ])
        else:
            self.overclustering_heads = None

        self.encoder_outputs = None
        self.head_outputs = None
        self.overclustering_head_outputs = None

    @property
    def eval_tensors(self):
        return th.cat(self.head_outputs, dim=1)

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.head_outputs = [nn.functional.softmax(self.clustering_head(x), dim=1) for x in self.encoder_outputs]

        if self.overclustering_heads is not None:
            self.overclustering_head_outputs = []
            for oc_head in self.overclustering_heads:
                oc_head_outs = [nn.functional.softmax(oc_head(x), dim=1) for x in self.encoder_outputs]
                self.overclustering_head_outputs.append(oc_head_outs)

        return self.dummy_output(views[0].size(0))
