import pickle
import torch as th
from torch import nn

import config
import helpers
from models.base.base_model_pretrain import BaseModelPreTrain
from models.base.base_model_kmeans import BaseModelKMeans
from lib.encoder import EncoderList
from lib.kernel import cdist
from register import register_model


@register_model
class SiameseNet(BaseModelKMeans, BaseModelPreTrain):
    def __init__(self, cfg):
        super(SiameseNet, self).__init__(cfg)

        head_configs = cfg.head_configs
        if not isinstance(head_configs, list):
            # Use the same head config for all views if we just got one.
            head_configs = cfg.n_views * [head_configs]
        self.heads = EncoderList(head_configs, input_sizes=self.encoders.output_sizes)

        self.encoder_outputs = None
        self.head_outputs = None
        self.pair_labels = None

    @property
    def eval_tensors(self):
        return th.stack([e[:, 0] for e in self.head_outputs], dim=1)

    def forward(self, views=None, paired_views=None, pair_labels=None):
        self.pair_labels = pair_labels

        if views is not None:
            # Regular forward. Compute embeddings
            self.encoder_outputs = self.encoders(views)
            self.head_outputs = self.heads(self.encoder_outputs)
        elif paired_views is not None:
            # `views` consists of pairs, so it has to be reshaped before and after embedding.
            batch_size = paired_views[0].size(0)
            inputs = [th.cat([v[:, 0], v[:, 1]], dim=0) for v in paired_views]
            encoder_outs = self.encoders(inputs)
            head_outs = self.heads(encoder_outs)

            self.encoder_outputs = [th.stack([o[:batch_size], o[batch_size:]], dim=1) for o in encoder_outs]
            self.head_outputs = [th.stack([o[:batch_size], o[batch_size:]], dim=1) for o in head_outs]
        else:
            raise RuntimeError()

        return self.head_outputs

    def split_batch(self, batch, **_):
        assert len(batch) == (2 * self.n_views + 1), f"Invalid number of tensors in batch ({len(batch)}) for model " \
                                                 f"{self.__class__.__name__}"
        paired_views = batch[:self.n_views]
        pair_labels = batch[self.n_views: (2 * self.n_views)]
        labels = batch[-1]
        return paired_views, pair_labels, labels

    def training_step(self, batch, idx):
        paired_views, pair_labels, _ = self.split_batch(batch)
        _ = self(paired_views=paired_views, pair_labels=pair_labels)
        losses = self.get_loss()
        self._log_dict(losses, prefix=f"train_loss")
        return losses["tot"]

    def _val_test_step(self, batch, idx, prefix):
        paired_views, pair_labels, labels = self.split_batch(batch)
        _ = self(paired_views=paired_views, pair_labels=pair_labels)

        losses = self.get_loss()
        self._log_dict(losses, prefix=f"{prefix}_loss")

        return helpers.npy(labels), helpers.npy(self.eval_tensors)


class Orthogonalization(nn.Module):
    def __init__(self, input_size):
        super(Orthogonalization, self).__init__()

        assert len(input_size) == 1
        self.register_buffer("ort_weights", th.eye(input_size[0]).to(device=config.DEVICE))
        self.small_number = 1e-3

    @th.no_grad()
    def update_ort_weights(self, ort):
        self.ort_weights = ort

    def forward(self, inp):
        if self.training:
            with th.no_grad():
                # Ensure input to Cholesky is PD
                cholesky_inp = inp.T @ inp + self.small_number * th.eye(inp.size(1)).type_as(inp)
                ort = th.linalg.cholesky(cholesky_inp)
                ort = th.linalg.inv(ort)
            self.update_ort_weights(ort)
        else:
            ort = self.ort_weights

        out = inp @ ort.T
        return out


class Affinity(nn.Module):
    def __init__(self, sigma, n_neighbors, n_scale_neighbors):
        super(Affinity, self).__init__()
        self.sigma = sigma
        self.n_neighbors = n_neighbors
        self.n_scale_neighbors = n_scale_neighbors
        self.big_number = 1e9

    def get_sigma(self, top_k_dists):
        if self.sigma is not None:
            return self.sigma

        # Median distance to 'scale_neighbors'-th neighbor
        ngh_dists = top_k_dists[:, self.n_scale_neighbors]
        sigma = th.sqrt(th.median(ngh_dists)).detach()
        return sigma

    def forward(self, inp):
        batch_size = inp.size(0)
        dist = cdist(inp, inp)
        dist.fill_diagonal_(self.big_number)
        top_k_dists, idx = th.topk(dist, k=self.n_neighbors, dim=1, largest=False, sorted=True)

        rang = th.arange(batch_size).long()
        rang = rang[:, None].expand(-1, self.n_neighbors)
        ngh_mask = th.zeros_like(dist).type_as(dist)
        ngh_mask[rang.reshape(-1, 1), idx.reshape(1, -1)] = 1

        sigma = self.get_sigma(top_k_dists)
        aff = th.exp(-1 * dist / (2 * sigma ** 2)) * ngh_mask.detach()
        sym = (aff + aff.T) / 2
        return sym


@register_model
class MvSCN(BaseModelPreTrain, BaseModelKMeans):
    def __init__(self, cfg):
        super(MvSCN, self).__init__(cfg)

        self.siamese_encoders = self.load_siamese_encoders(dir_name=cfg.siam_dir, ckpt=cfg.siam_ckpt)
        self.affinity_modules = EncoderList(encoder_modules=[
            Affinity(sigma=cfg.aff_sigma, n_neighbors=cfg.aff_n_neighbors, n_scale_neighbors=cfg.aff_n_scale_neighbors)
            for _ in range(cfg.n_views)
        ])

        head_configs = cfg.head_configs
        if not isinstance(head_configs, list):
            # Use the same head config for all views if we just got one.
            head_configs = cfg.n_views * [head_configs]
        self.heads = EncoderList(head_configs, input_sizes=self.encoders.output_sizes)

        self.orthogonalize = EncoderList(encoder_modules=[
            Orthogonalization(input_size=self.heads.output_sizes[v]) for v in range(cfg.n_views)
        ])
        
        self.siam_outputs = None
        self.affinities = None
        self.encoder_outputs = None
        self.head_outputs = None
        self.orthogonalized = None

    @staticmethod
    def load_siamese_encoders(dir_name, ckpt):
        model_dir = config.MODELS_DIR / dir_name
        with open(model_dir / "config.pkl", "rb") as f:
            siam_cfg = pickle.load(f)

        siam_model = SiameseNet.load_from_checkpoint(str(model_dir / ckpt), cfg=siam_cfg.model_config)

        encoders = []
        for enc, head in zip(siam_model.encoders, siam_model.heads):
            encoders.append(nn.Sequential(enc, head))

        encoders = EncoderList(encoder_modules=encoders)

        for param in encoders.parameters():
            param.requires_grad = False
        return encoders

    @property
    def eval_tensors(self):
        return th.cat(self.orthogonalized, dim=1)

    def forward(self, views):
        with th.no_grad():
            self.siam_outputs = self.siamese_encoders(views)
            self.affinities = self.affinity_modules(self.siam_outputs)

        self.encoder_outputs = self.encoders(views)
        self.head_outputs = self.heads(self.encoder_outputs)
        self.orthogonalized = self.orthogonalize(self.head_outputs)

        return self.dummy_output(views[0].size(0))
