import torch as th
from torch import nn

import config
from models.base.base_model_pretrain import BaseModelPreTrain
from models.base.base_model_spectral import BaseModelSpectral
from lib.encoder import EncoderList
from register import register_model


class SelfExpressiveLayer(nn.Module):
    def __init__(self, n_samples):
        super(SelfExpressiveLayer, self).__init__()

        self.n_samples = n_samples
        initial_weights = th.full((n_samples, n_samples), 1e-4) - (1e-4 * th.eye(n_samples))
        self.register_parameter(
            "weight", nn.Parameter(data=initial_weights.to(device=config.DEVICE), requires_grad=True)
        )

    def weight_zero_diag(self):
        return self.weight - th.eye(self.n_samples).type_as(self.weight) * self.weight

    def forward(self, inp):
        w = self.weight_zero_diag()
        return w @ inp


@register_model
class DMSC(BaseModelPreTrain, BaseModelSpectral):
    def __init__(self, cfg):
        super(DMSC, self).__init__(cfg, flatten_encoder_output=False)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes)
        self.self_expressive = SelfExpressiveLayer(n_samples=cfg.n_samples)
        
        self.calc_self_representations = True
        self.views = None
        self.encoder_outputs = None
        self.latents = None
        self.decoder_outputs = None
        self.self_representations = None

    @property
    def affinity(self):
        abs_w = th.abs(self.self_expressive.weight_zero_diag())
        return (abs_w + abs_w.T) / 2

    def init_pre_train(self):
        super(DMSC, self).init_pre_train()
        self.calc_self_representations = False

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.decoder_outputs = self.decoders(self.encoder_outputs)

        if self.calc_self_representations:
            self.latents = [th.flatten(out, start_dim=1) for out in self.encoder_outputs]
            self.self_representations = [self.self_expressive(lat) for lat in self.latents]
        else:
            self.latents = self.self_representations = None

        return self.dummy_output(views[0].size(0))
