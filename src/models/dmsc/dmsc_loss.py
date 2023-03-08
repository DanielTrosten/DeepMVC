import torch as th
from torch import nn

import config
from lib.loss.terms import LossTerm
from register import register_loss_term


@register_loss_term
class DMSC1(LossTerm):
    def forward(self, net, cfg, extra):
        w = net.self_expressive.weight_zero_diag()
        return th.sqrt(th.sum(w ** 2))


@register_loss_term
class DMSC2(LossTerm):
    def forward(self, net, cfg, extra):
        losses = {}
        for v in range(cfg.n_views):
            dif = net.latents[v] - net.self_representations[v]
            losses[str(v)] = (dif ** 2).sum() / 2
        return losses


@register_loss_term
class DMSC3(LossTerm):
    def forward(self, net, cfg, extra):
        losses = {}
        for v in range(cfg.n_views):
            dif = net.views[v] - net.decoder_outputs[v]
            losses[str(v)] = th.sqrt(th.sum(dif ** 2)) / 2
        return losses
