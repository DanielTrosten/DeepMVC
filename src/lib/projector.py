import torch as th
import torch.nn as nn
from copy import deepcopy

from lib.encoder import EncoderList, Encoder


class IdentityProjector(nn.Module):
    @staticmethod
    def forward(inputs):
        return inputs


class Projector(nn.Module):
    def __init__(self, cfg, input_sizes):
        super(Projector, self).__init__()

        self.output_sizes = deepcopy(input_sizes)

        if cfg is None:
            self.op = IdentityProjector()
            self._forward = self.op

        elif cfg.layout == "separate":
            encoder_config = cfg.encoder_config
            if not isinstance(encoder_config, (list, tuple)):
                # If we didn't get a list of configs (one for each view), then duplicate the one we got across all views
                encoder_config = len(input_sizes) * [encoder_config]

            self.op = EncoderList(encoder_config, input_sizes=input_sizes)
            self._forward = self._list_forward
            self.output_sizes = self.op.output_sizes

        elif cfg.layout == "shared":
            assert all([input_sizes[0] == s for s in input_sizes]), "Shared projection head assumes that all encoder " \
                                                                    "output sizes are equal"
            self.op = Encoder(cfg.encoder_config, input_size=input_sizes[0])
            self._forward = self._concat_forward
            self.output_sizes = len(input_sizes) * [self.op.output_size]

        else:
            raise ValueError(f"Invalid projector layout: {cfg.layout}")

    def _list_forward(self, views):
        return self.op(views)

    def _concat_forward(self, views):
        v, n = len(views), views[0].size(0)
        projections = self.op(th.cat(views, dim=0))
        projections = [projections[idx] for idx in th.arange(n * v).view(v, n)]
        return projections

    def forward(self, views):
        return self._forward(views)
