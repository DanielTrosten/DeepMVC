import sys
import torch as th
from torch import nn

from lib.loss.terms import LossTerm
from lib.loss.utils import at_least_epsilon
from register import register_loss_term


@register_loss_term
class MIContrast(LossTerm):
    """Adapted from https://github.com/XLearning-SCU/2021-CVPR-Completer/blob/main/loss.py"""
    def __init__(self, cfg, params, net):
        super(MIContrast, self).__init__(cfg, params, net)

        self.alpha = cfg.alpha
        self.eps = sys.float_info.epsilon

    @staticmethod
    def compute_joint(z1, z2):
        """Compute the joint probability matrix P"""

        bn, k = z1.size()
        assert (z2.size(0) == bn and z2.size(1) == k)

        p_i_j = z1.unsqueeze(2) * z2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2  # symmetrize
        p_i_j = p_i_j / p_i_j.sum()      # normalise

        return p_i_j

    def _loss(self, z1, z2):
        """Contrastive loss for maximizing the consistency"""
        _, k = z1.size()
        p_i_j = self.compute_joint(z1, z2)
        assert (p_i_j.size() == (k, k))

        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

        p_i_j = at_least_epsilon(p_i_j)
        p_j = at_least_epsilon(p_j)
        p_i = at_least_epsilon(p_i)

        loss = - p_i_j * (th.log(p_i_j) - (self.alpha + 1) * th.log(p_j) - (self.alpha + 1) * th.log(p_i))
        loss = loss.sum()
        return loss

    def forward(self, net, cfg, extra):
        losses = {}
        for u in range(cfg.n_views - 1):
            for v in range(u, cfg.n_views):
                losses[f"{u}{v}"] = self._loss(net.encoder_outputs[u], net.encoder_outputs[v])
        return losses

