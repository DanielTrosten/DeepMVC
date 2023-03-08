import numpy as np
import torch as th
from torch import nn

import config
from lib.loss.terms import LossTerm
from register import register_loss_term


# ======================================================================================================================
# Utility functions
# Adapted from: https://github.com/SubmissionsIn/Multi-VAE/blob/main/multi_vae/MvTraining.py
# ======================================================================================================================

def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mean**2 - th.exp(logvar))
    # Mean KL divergence across batch for each latent variable
    kl_means = th.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = th.sum(kl_means)

    return kl_loss


def _kl_discrete_loss(alpha, eps=1e-12):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : th.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    log_dim = th.Tensor([np.log(disc_dim)]).type_as(alpha)

    # Calculate negative entropy of each row
    neg_entropy = th.sum(alpha * th.log(alpha + eps), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = th.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss


# ======================================================================================================================
# Loss terms
# ======================================================================================================================

@register_loss_term
class MVAECont(LossTerm):
    @staticmethod
    def _loss(means, log_vars, cap):
        kl_loss = _kl_normal_loss(means, log_vars)
        loss = th.abs(cap - kl_loss)
        return loss

    def forward(self, net, cfg, extra):
        cap_step = cfg.cont_max_capacity / cfg.iters_add_capacity
        current_cap = min(net.current_train_step * cap_step, cfg.cont_max_capacity)

        losses = {}
        for v in range(cfg.n_views):
            losses[str(v)] = self._loss(net.means[v], net.log_vars[v], current_cap) / net.n_pixels[v]

        return losses


@register_loss_term
class MVAEDisc(LossTerm):
    @staticmethod
    def _loss(alpha, cap):
        kl_loss = _kl_discrete_loss(alpha)
        loss = th.abs(cap - kl_loss)
        return loss

    def forward(self, net, cfg, extra):
        max_cap = np.log(cfg.n_clusters)
        cap_step = max_cap / cfg.iters_add_capacity
        current_cap = min(net.current_train_step * cap_step, max_cap)
        loss = cfg.n_views * self._loss(net.alpha, current_cap) / np.sum(net.n_pixels)
        return loss


@register_loss_term
class MVAERec(LossTerm):
    @staticmethod
    def _loss(rec, view):
        return nn.functional.binary_cross_entropy(input=rec, target=view, reduction="mean")

    def forward(self, net, cfg, extra):
        losses = {}
        batch_size = net.views[0].size(0)
        for v in range(cfg.n_views):
            view = net.views[v].view(batch_size, -1)
            rec = net.decoder_outputs[v].view(batch_size, -1)
            losses[str(v)] = self._loss(rec, view) * net.n_pixels[v]

        return losses
