import torch as th
from torch import nn

from lib.encoder import Encoder, EncoderList
from lib.fusion import get_fusion_module
from models.clustering_module import get_clustering_module
from models.clustering_module import kernel_width
from models.base.base_model_pretrain import BaseModelPreTrain
from register import register_model


class Discriminator(nn.Module):
    def __init__(self, cfg, input_size):
        super(Discriminator, self).__init__()
        self.mlp = Encoder(cfg.mlp_config, input_size=input_size)
        self.output_layer = nn.Sequential(
            nn.Linear(self.mlp.output_size[0], 1, bias=True),
            nn.Sigmoid()
        )
        self.d0 = self.dv = None

    def forward(self, x0, xv):
        self.d0 = self.output_layer(self.mlp(x0))
        self.dv = self.output_layer(self.mlp(xv))
        return [self.d0, self.dv]


class AttentionLayer(nn.Module):
    def __init__(self, cfg, input_size):
        super(AttentionLayer, self).__init__()
        self.tau = cfg.tau
        self.mlp = Encoder(cfg.mlp_config, input_size=[input_size[0] * cfg.n_views])
        self.output_layer = nn.Linear(self.mlp.output_size[0], cfg.n_views, bias=True)
        self.weights = None

    def forward(self, xs):
        h = th.cat(xs, dim=1)
        act = self.output_layer(self.mlp(h))
        e = nn.functional.softmax(th.sigmoid(act) / self.tau, dim=1)
        self.weights = th.mean(e, dim=0)
        return self.weights


class DummyAttentionLayer(nn.Module):
    @staticmethod
    def forward(xs):
        n_views = len(xs)
        weights = th.ones(n_views).type_as(xs[0]) / n_views
        return weights


@register_model
class EAMC(BaseModelPreTrain):
    def __init__(self, cfg):
        super(EAMC, self).__init__(cfg)

        assert all([self.encoders.output_sizes[0] == s for s in self.encoders.output_sizes])
        hidden_size = self.encoders.output_sizes[0]

        if cfg.attention_config is not None:
            self.attention = AttentionLayer(cfg.attention_config, input_size=hidden_size)
            self.fusion = None
            fused_size = hidden_size
        else:
            self.attention = None
            assert cfg.fusion_config is not None, "EAMC expects either attention_config or fusion_config to be not None"
            self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
            fused_size = self.fusion.output_size

        self.discriminators = nn.ModuleList(
            [Discriminator(cfg.discriminator_config, input_size=hidden_size)
             for _ in range(len(cfg.encoder_configs) - 1)]
            )

        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=fused_size)

        # Kernel widths for view-specific and fused representations
        self.encoder_kernel_width = kernel_width.get_kernel_width_module(
            cfg.encoder_kernel_width_config, input_size=None
        )
        self.fused_kernel_width = kernel_width.get_kernel_width_module(
            cfg.fused_kernel_width_config, input_size=None
        )

        self.encoder_outputs = None
        self.discriminator_outputs = None
        self.weights = None
        self.fused = None
        self.hidden = None
        self.output = None

    def configure_optimizers(self):
        # Optimizer for encoders, attention and clustering module
        groups = [
            {'params': self.encoders.parameters(), 'lr': 1e-5},
            {'params': self.clustering_module.parameters(), 'lr': 1e-5}
        ]
        if self.attention is not None:
            groups.append({'params': self.attention.parameters(), 'lr': 1e-4})
        else:
            groups.append({'params': self.fusion.parameters(), 'lr': 1e-4})

        enc = th.optim.Adam(groups, betas=(0.95, 0.999))
        # Optimizer for discriminator
        disc = th.optim.Adam(self.discriminators.parameters(), 1e-3, betas=(0.5, 0.999))
        return enc, disc

    def training_step(self, batch, batch_idx, optimizer_idx):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)
        losses = self.get_loss()

        if optimizer_idx == 0:
            # Train encoders, attention and clustering module.
            del losses["EAMCDiscriminator"]
            del losses["tot"]
        elif optimizer_idx == 1:
            # Train discriminator
            losses = {"EAMCDiscriminator": losses["EAMCDiscriminator"]}
        else:
            raise RuntimeError()

        losses["tot"] = sum(losses.values())
        self._log_dict(losses, prefix="train_loss")
        return losses["tot"]

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.discriminator_outputs = [
            self.discriminators[i](self.encoder_outputs[0], self.encoder_outputs[i + 1])
            for i in range(len(self.encoder_outputs) - 1)
        ]

        if self.attention is not None:
            self.weights = self.attention(self.encoder_outputs)
            self.fused = th.sum(self.weights[None, None, :] * th.stack(self.encoder_outputs, dim=-1), dim=-1)
        else:
            self.fused = self.fusion(self.encoder_outputs)

        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output
