import torch as th
from torch import nn

import config
from lib.loss.terms import LossTerm
from register import register_loss_term


@register_loss_term
class Contrast(LossTerm):
    def __init__(self, cfg, params, net):
        super(Contrast, self).__init__(cfg, params, net)
        self.tau = cfg.tau
        self.n_views = cfg.n_views
        self.adaptive_weight = getattr(cfg, "contrast_adaptive_weight", False)
        self.large_num = 1e9
        # TODO: Sample negatives from other clusters

    def _contrastive_loss_two_views(self, h1, h2):
        """
        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py
        """
        n = h1.size(0)
        labels = th.arange(0, n, device=config.DEVICE, dtype=th.long)
        masks = th.eye(n, device=config.DEVICE)

        logits_aa = ((h1 @ h1.t()) / self.tau) - masks * self.large_num
        logits_bb = ((h2 @ h2.t()) / self.tau) - masks * self.large_num

        logits_ab = (h1 @ h2.t()) / self.tau
        logits_ba = (h2 @ h1.t()) / self.tau

        loss_a = th.nn.functional.cross_entropy(th.cat((logits_ab, logits_aa), dim=1), labels)
        loss_b = th.nn.functional.cross_entropy(th.cat((logits_ba, logits_bb), dim=1), labels)

        loss = (loss_a + loss_b)
        return loss

    @th.no_grad()
    def _get_weight(self, net):
        if self.adaptive_weight:
            weights = nn.functional.softmax(net.fusion.weights, dim=-1)
            weight = weights.min().detach()
        else:
            weight = 1.0
        return weight

    def forward(self, net, cfg, extra):
        z = th.stack(net.projections, dim=0)
        z = nn.functional.normalize(z, dim=-1, p=2)
        # sim = th.einsum("uid,vjd->uvij", z, z) / self.tau
        weight = self._get_weight(net)
        weight *= 2 / (cfg.n_views * (cfg.n_views - 1))
        losses = {}
        for u in range(self.n_views - 1):
            for v in range(u + 1, self.n_views):
                losses[f"{u}{v}"] = weight * self._contrastive_loss_two_views(z[u], z[v])

        return losses


# class _Contrast(LossTerm):
#     def __init__(self, cfg, params, net):
#         super(_Contrast, self).__init__(cfg, params, net)
#         self.tau = cfg.tau
#         self.n_views = cfg.n_views
#         self.adaptive_weight = getattr(cfg, "contrast_adaptive_weight", False)
#         # TODO: Sample negatives from other clusters
#
#     def _contrastive_loss_two_views(self, sim):
#         labels = th.arange(sim.size(0)).to(device=config.DEVICE)
#         loss = nn.functional.cross_entropy(input=sim, target=labels)
#         return loss
#
#     @th.no_grad()
#     def _get_weight(self, net):
#         if self.adaptive_weight:
#             weights = nn.functional.softmax(net.fusion.weights)
#             weight = weights.min().detach()
#         else:
#             weight = 1.0
#         return weight
#
#     def forward(self, net, cfg, extra):
#         z = th.stack(net.projections, dim=0)
#         z = nn.functional.normalize(z, dim=-1, p=2)
#         sim = th.einsum("uid,vjd->uvij", z, z) / self.tau
#         weight = self._get_weight(net)
#         losses = {}
#         for u in range(self.n_views):
#             for v in range(self.n_views):
#                 if u != v:
#                     losses[f"{u}{v}"] = weight * self._contrastive_loss_two_views(sim[u, v, :, :])
#
#         return losses
