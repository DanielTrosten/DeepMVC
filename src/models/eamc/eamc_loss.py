import torch as th
from torch.nn.functional import binary_cross_entropy

import config
from lib.loss.terms import LossTerm
from register import register_loss_term


@register_loss_term
class EAMCAttention(LossTerm):
    required_tensors = ["encoder_kernels", "fused_kernel"]

    def forward(self, net, cfg, extra):
        kc = th.sum(net.weights[None, None, :] * th.stack(extra["encoder_kernels"], dim=-1), dim=-1)
        dif = (extra["fused_kernel"] - kc)
        return th.trace(dif @ th.t(dif))


@register_loss_term
class EAMCGenerator(LossTerm):
    def forward(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        target = th.ones(net.output.size(0), device=config.DEVICE)
        for _, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), target)
            # tot += th.mean(th.log(d0 + loss.EPSILON)) + th.mean(th.log(1 - dv + loss.EPSILON))
            # tot += th.mean(-1 * th.log(dv + loss.EPSILON))
        return tot


@register_loss_term
class EAMCDiscriminator(LossTerm):
    def forward(self, net, cfg, extra):
        tot = th.tensor(0., device=config.DEVICE)
        real_target = th.ones(net.output.size(0), device=config.DEVICE)
        fake_target = th.zeros(net.output.size(0), device=config.DEVICE)
        for d0, dv in net.discriminator_outputs:
            tot += binary_cross_entropy(dv.squeeze(), fake_target) + binary_cross_entropy(d0.squeeze(), real_target)
            # tot += th.mean(th.log(d0 + loss.EPSILON)) + th.mean(th.log(1 - dv + loss.EPSILON))
        return tot

