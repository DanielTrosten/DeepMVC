import numpy as np
import torch as th
import torch.nn as nn

from lib import kernel
from lib.loss.utils import d_cs, triu, at_least_epsilon
from register import register_loss_term


# ======================================================================================================================
# Generic loss terms
# ======================================================================================================================

class LossTerm(nn.Module):
    # Names of tensors required for the loss computation
    required_tensors = []

    def __init__(self, cfg, params, net):
        """
        Base class for a term in the loss function.
        """
        super(LossTerm, self).__init__()

    def forward(self, net, cfg, extra):
        raise NotImplementedError()


@register_loss_term
class DDC1(LossTerm):
    """
    L_1 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def forward(self, net, cfg, extra):
        numerator_epsilon = getattr(cfg, "numerator_epsilon", True)
        return d_cs(net.output, extra["hidden_kernel"], cfg.n_clusters, numerator_epsilon=numerator_epsilon)


@register_loss_term
class DDC2(LossTerm):
    """
    L_2 loss from DDC
    """
    def forward(self, net, cfg, extra):
        n = net.output.size(0)
        return 2 / (n * (n - 1)) * triu(net.output @ th.t(net.output))


@register_loss_term
class DDC2Flipped(LossTerm):
    """
    Flipped version of the L_2 loss from DDC. Used by EAMC
    """

    def forward(self, net, cfg, extra):
        return 2 / (cfg.n_clusters * (cfg.n_clusters - 1)) * triu(th.t(net.output) @ net.output)


@register_loss_term
class DDC3(LossTerm):
    """
    L_3 loss from DDC
    """
    required_tensors = ["hidden_kernel"]

    def forward(self, net, cfg, extra):
        eye = th.eye(cfg.n_clusters).type_as(net.output)
        m = th.exp(-kernel.cdist(net.output, eye))
        numerator_epsilon = getattr(cfg, "numerator_epsilon", True)
        return d_cs(m, extra["hidden_kernel"], cfg.n_clusters, numerator_epsilon=numerator_epsilon)


@register_loss_term
class MSE(LossTerm):
    def forward(self, net, cfg, extra):
        losses = {}
        for v, (inp, rec) in enumerate(zip(net.views, net.decoder_outputs)):
            losses[str(v)] = nn.functional.mse_loss(input=rec, target=inp.view(rec.size()))
        return losses


@register_loss_term
class Zero(LossTerm):
    """ Dummy loss. Returns a constant 0. """
    def forward(self, net, cfg, extra):
        return th.tensor(0.0)
