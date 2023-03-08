import numpy as np
import torch as th
from torch import nn

from models.base.base_model_pretrain import BaseModelPreTrain
from models.base.base_model_kmeans import BaseModelKMeans
from lib.encoder import EncoderList
from register import register_model


@register_model
class MultiVAE(BaseModelPreTrain, BaseModelKMeans):
    def __init__(self, cfg):
        super(MultiVAE, self).__init__(cfg)

        self.current_train_step = 0
        self.eps = 1e-12
        self.temperature = cfg.temperature
        self.n_clusters = cfg.n_clusters
        self.cont_dim = cfg.cont_dim
        self.n_views = cfg.n_views
        self.hidden_dim = cfg.hidden_dim
        self.latent_dim = self.cont_dim + self.n_clusters
        self.n_pixels = [np.prod(e.input_size) for e in cfg.encoder_configs]

        self.features_to_hidden = EncoderList(encoder_modules=[
            nn.Sequential(
                nn.Linear(in_features=self.encoders.output_sizes[v][0], out_features=self.hidden_dim),
                nn.ReLU()
            ) for v in range(self.n_views)])

        self.mean_transforms = EncoderList(
            encoder_modules=
            [nn.Linear(in_features=self.hidden_dim, out_features=self.cont_dim) for _ in range(self.n_views)]
        )
        self.log_var_transforms = EncoderList(
            encoder_modules=
            [nn.Linear(in_features=self.hidden_dim, out_features=self.cont_dim) for _ in range(self.n_views)]
        )
        self.alpha_transform = nn.Sequential(
            nn.Linear(in_features=(self.hidden_dim * self.n_views), out_features=self.n_clusters),
            nn.Softmax(dim=-1)
        )

        self.latents_to_features = EncoderList(encoder_modules=[
            nn.Sequential(
                nn.Linear(in_features=self.latent_dim, out_features=self.hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=self.hidden_dim, out_features=self.encoders.output_sizes[v][0]),
                nn.ReLU(),
            ) for v in range(self.n_views)])

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes_before_flatten)

        self.hidden_size = self.encoders.output_sizes[0]

        self.views = None
        self.encoder_outputs = None
        self.hidden = None
        self.fused = None
        self.means = None
        self.log_vars = None
        self.alpha = None
        self.latent_cont_samples = None
        self.latent_disc_samples = None
        self.decoder_latents = None
        self.decoder_features = None
        self.decoder_outputs = None

    @property
    def eval_tensors(self):
        return th.cat([self.latent_disc_samples] + self.latent_cont_samples, dim=1)

    def sample_normal(self, mean, log_var):
        """
        https://github.com/SubmissionsIn/Multi-VAE/blob/main/multi_vae/MvModels.py
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.
        log_var : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = th.exp(0.5 * log_var)
            eps = th.zeros(std.size()).normal_().type_as(mean)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        https://github.com/SubmissionsIn/Multi-VAE/blob/main/multi_vae/MvModels.py
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : th.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = th.rand(alpha.size()).type_as(alpha)
            gumbel = -th.log(-th.log(unif + self.eps) + self.eps)
            # Reparameterize to create gumbel softmax sample
            log_alpha = th.log(alpha + self.eps)
            logit = (log_alpha + gumbel) / self.temperature
            return nn.functional.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = th.max(alpha, dim=1)
            one_hot_samples = th.zeros(alpha.size()).type_as(alpha)
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data, 1)
            return one_hot_samples

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.hidden = self.features_to_hidden(self.encoder_outputs)

        # self.fused = th.cat(self.hidden, dim=1)
        self.means = self.mean_transforms(self.hidden)
        self.log_vars = self.log_var_transforms(self.hidden)
        self.alpha = self.alpha_transform(th.cat(self.hidden, dim=1))

        self.latent_cont_samples = [self.sample_normal(self.means[i], self.log_vars[i]) for i in range(self.n_views)]
        self.latent_disc_samples = self.sample_gumbel_softmax(self.alpha)

        self.decoder_features = []
        for v in range(self.n_views):
            lat = th.cat([self.latent_cont_samples[v], self.latent_disc_samples], dim=1)
            feat = self.latents_to_features[v](lat).view(-1, *self.encoders.output_sizes_before_flatten[v])
            self.decoder_features.append(feat)

        self.decoder_outputs = self.decoders(self.decoder_features)
        
        return self.dummy_output(views[0].size(0))

    def training_step(self, batch, idx):
        self.current_train_step += 1
        return super(MultiVAE, self).training_step(batch, idx)
        # *inputs, labels = self.split_batch(batch, includes_labels=True)
        # _ = self(*inputs)
        # losses = self.get_loss()
        # self._log_dict(losses, prefix="train_loss")
        # return losses["tot"]
