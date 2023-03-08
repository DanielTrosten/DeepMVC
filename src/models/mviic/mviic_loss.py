import sys
from abc import ABC
import torch as th

from register import register_loss_term
from lib.loss.terms import LossTerm
from lib.loss.utils import at_least_epsilon


class BaseIIC(ABC, LossTerm):
    """Adapted from https://github.com/XLearning-SCU/2021-CVPR-Completer/blob/main/loss.py"""

    def __init__(self, cfg, params, net):
        super(BaseIIC, self).__init__(cfg, params, net)

        self.lam = cfg.lam
        self.eps = sys.float_info.epsilon

    @staticmethod
    def compute_joint(z1, z2):
        """Compute the joint probability matrix P"""

        bn, k = z1.size()
        assert (z2.size(0) == bn and z2.size(1) == k)

        p_i_j = z1.unsqueeze(2) * z2.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        p_i_j = (p_i_j + p_i_j.t()) / 2  # symmetrize
        p_i_j = p_i_j / p_i_j.sum()  # normalise

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

        loss = - p_i_j * (th.log(p_i_j) - self.lam * th.log(p_j) - self.lam * th.log(p_i))
        loss = loss.sum()
        return loss


@register_loss_term
class IICClustering(BaseIIC):
    def forward(self, net, cfg, extra):
        predictions = net.head_outputs
        losses = {}
        for u in range(cfg.n_views - 1):
            for v in range(u+1, cfg.n_views):
                losses[f"{u}{v}"] = self._loss(predictions[u], predictions[v])
        return losses


@register_loss_term
class IICOverClustering(BaseIIC):
    def forward(self, net, cfg, extra):
        losses = {}
        for u in range(cfg.n_views - 1):
            for v in range(u+1, cfg.n_views):
                head_outputs = net.overclustering_head_outputs
                n_heads = len(head_outputs)

                # Average over losses for each overclustering head.
                _losses = [
                    (self._loss(oc_head_out[u], oc_head_out[v]) / n_heads) for oc_head_out in head_outputs
                ]
                losses[f"{u}{v}"] = sum(_losses)
        return losses
