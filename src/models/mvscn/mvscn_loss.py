from torch import nn

from lib.loss.terms import LossTerm
from lib.kernel import cdist
from register import register_loss_term


@register_loss_term
class Siamese(LossTerm):
    @staticmethod
    def _loss(pairs, pair_labels):
        dists = ((pairs[:, 0] - pairs[:, 1])**2).sum(dim=1)
        term_1 = pair_labels * dists
        term_2 = (1 - pair_labels) * nn.functional.relu(1 - dists)
        loss = (term_1 + term_2).mean()
        return loss

    def forward(self, net, cfg, extra):
        losses = {}
        for v in range(cfg.n_views):
            losses[str(v)] = self._loss(net.encoder_outputs[v], net.pair_labels[v])
        return losses


@register_loss_term
class MVSCN1(LossTerm):
    @staticmethod
    def _loss(aff, ort):
        dists = cdist(ort, ort)
        loss = (dists * aff).mean()
        return loss

    def forward(self, net, cfg, extra):
        losses = {}
        for v in range(cfg.n_views):
            losses[str(v)] = self._loss(net.affinities[v], net.orthogonalized[v])
        return losses


@register_loss_term
class MVSCN2(LossTerm):
    @staticmethod
    def _loss(ort_1, ort_2):
        return ((ort_1 - ort_2)**2).sum(dim=1).mean()

    def forward(self, net, cfg, extra):
        losses = {}
        for v in range(cfg.n_views - 1):
            for u in range(v + 1, cfg.n_views):
                losses[f"{u}{v}"] = 2 / (cfg.n_views ** 2) * self._loss(net.orthogonalized[v], net.orthogonalized[u])
        return losses
