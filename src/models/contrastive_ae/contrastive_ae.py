import torch as th
from torch import nn

from lib.fusion import get_fusion_module
from lib.encoder import EncoderList, Encoder
from lib.normalization import get_normalizer
from models.clustering_module import get_clustering_module
from models.base import BaseModelKMeans, BaseModelPreTrain
from register import register_model


@register_model
class CAE(BaseModelPreTrain):
    def __init__(self, cfg):
        super(CAE, self).__init__(cfg)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes_before_flatten)

        self.fusion = get_fusion_module(cfg.fusion_config, input_sizes=self.encoders.output_sizes)
        self.clustering_module = get_clustering_module(cfg.cm_config, input_size=self.fusion.output_size)

        if cfg.projector_config is not None:
            self.projector = Encoder(cfg.projector_config)
        else:
            self.projector = nn.Identity()

        self.views = None
        self.encoder_outputs = None
        self.decoder_outputs = None
        self.projections = None
        self.fused = None
        self.hidden = None
        self.output = None

    @property
    def fusion_weights(self):
        w = getattr(self.fusion, "weights", None)
        if w is not None:
            w = nn.functional.softmax(w.squeeze(), dim=-1).detach()
        return w

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.projections = [self.projector(x) for x in self.encoder_outputs]
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)

        self.decoder_outputs = self.decoders(
            [inp.view(-1, *size) for inp, size in zip(self.encoder_outputs, self.encoders.output_sizes_before_flatten)]
        )
        return self.output


@register_model
class CAEKM(BaseModelKMeans, BaseModelPreTrain):
    def __init__(self, cfg):
        super(CAEKM, self).__init__(cfg, flatten_encoder_output=False)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes)

        self.latent_normalizer = get_normalizer(getattr(cfg, "latent_norm", None))

        if cfg.projector_config is not None:
            self.projector = Encoder(cfg.projector_config)
        else:
            self.projector = nn.Identity()

        self.views = None
        self.encoder_outputs = None
        self.latents = None
        self.decoder_outputs = None
        self.projections = None

    @property
    def eval_tensors(self):
        return th.cat(self.latents, dim=1)

    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)
        self.latents = [self.latent_normalizer(th.flatten(x, start_dim=1)) for x in self.encoder_outputs]
        self.projections = [self.projector(lat) for lat in self.latents]
        self.decoder_outputs = self.decoders(self.encoder_outputs)

        return self.dummy_output(views[0].size(0))
