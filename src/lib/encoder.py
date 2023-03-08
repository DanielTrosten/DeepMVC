from copy import deepcopy
import numpy as np
import torch.nn as nn
from torchvision import models as torchvision_models

import helpers
from config.templates.layers import DEFAULT_LAYERS
from lib.normalization import get_normalizer


def _get_conv_padding(pad_mode, ksize):
    if pad_mode == "valid":
        return 0, 0
    if pad_mode == "same":
        ksize = helpers.ensure_iterable(ksize, expected_length=2)
        return ksize[0] // 2, ksize[1] // 2
    raise RuntimeError(f"Unknown padding mode: {pad_mode}.")


def _check_input_size(input_size, expected_len, layer_type):
    expected_len = helpers.ensure_iterable(expected_len, assert_length=False)
    assert len(input_size) in expected_len, f"Length of input size {input_size} is not in {expected_len} valid for " \
                                            f"layer '{layer_type}'."


def conv(cfg, input_size):
    _check_input_size(input_size, 3, "conv")
    pad = _get_conv_padding(cfg.padding, cfg.kernel_size)
    layer = nn.Conv2d(in_channels=input_size[0], out_channels=cfg.out_channels, kernel_size=cfg.kernel_size,
                      padding=pad)
    output_size = [cfg.out_channels, *helpers.conv2d_output_shape(input_size[1:], kernel_size=cfg.kernel_size, pad=pad)]
    return layer, output_size


def conv_transpose(cfg, input_size):
    _check_input_size(input_size, 3, "conv_transpose")
    pad = _get_conv_padding(cfg.padding, cfg.kernel_size)
    layer = nn.ConvTranspose2d(in_channels=input_size[0], out_channels=cfg.out_channels, kernel_size=cfg.kernel_size,
                               padding=pad)
    output_size = [cfg.out_channels,
                   *helpers.conv_transpose_2d_output_shape(input_size[1:], kernel_size=cfg.kernel_size, pad=pad)]
    return layer, output_size


def dense(cfg, input_size):
    _check_input_size(input_size, 1, "dense")
    if cfg.n_units == -1:
        n_units = input_size[0]
    else:
        n_units = cfg.n_units
    layer = nn.Linear(in_features=input_size[0], out_features=n_units, bias=cfg.bias)
    output_size = [n_units]
    return layer, output_size


def batch_normalization(cfg, input_size):
    _check_input_size(input_size, (1, 3), "batch_norm")

    if len(input_size) > 1:
        bn_layer_class = nn.BatchNorm2d
    else:
        bn_layer_class = nn.BatchNorm1d

    bn_layer = bn_layer_class(num_features=input_size[0], affine=cfg.affine)
    return bn_layer, input_size


def max_pool(cfg, input_size):
    _check_input_size(input_size, 3, "max_pool")
    layer = nn.MaxPool2d(kernel_size=cfg.kernel_size)
    output_size = [input_size[0], *helpers.conv2d_output_shape(input_size[1:], kernel_size=cfg.kernel_size,
                                                               stride=cfg.kernel_size)]
    return layer, output_size


def up_sample(cfg, input_size):
    _check_input_size(input_size, 3, "up_sample")
    layer = nn.Upsample(scale_factor=cfg.kernel_size, mode=cfg.mode)
    output_size = [input_size[0], int(cfg.kernel_size * input_size[1]), int(cfg.kernel_size * input_size[2])]
    return layer, output_size


def adaptive_avg_pool(cfg, input_size):
    _check_input_size(input_size, 3, "adaptive_avg_pool")
    layer = nn.AdaptiveAvgPool2d(output_size=cfg.output_size)
    output_size = [input_size[0], *cfg.output_size]
    return layer, output_size


def flatten(cfg, input_size):
    # _check_input_size(input_size, 3, "flatten")
    return nn.Flatten(start_dim=1), [np.prod(input_size)]


def activation(cfg, input_size):
    if cfg.activation == "relu":
        layer = nn.ReLU()
    elif cfg.activation == "sigmoid":
        layer = nn.Sigmoid()
    elif cfg.activation == "tanh":
        layer = nn.Tanh()
    elif cfg.activation == "softmax":
        layer = nn.Softmax(dim=-1)
    elif cfg.activation == "leaky_relu":
        layer = nn.LeakyReLU(cfg.activation_params["neg_slope"])
    else:
        raise RuntimeError(f"Invalid activation: {cfg.activation}.")
    return layer, input_size


class Encoder(nn.Module):
    layer_getters = {
        "Conv": conv,
        "ConvTranspose": conv_transpose,
        "Dense": dense,
        "MaxPool": max_pool,
        "UpSample": up_sample,
        "AdaptiveAvgPool": adaptive_avg_pool,
        "BatchNormalization": batch_normalization,
        "Activation": activation,
        "Flatten": flatten,
    }

    def __init__(self, cfg, input_size=None, flatten_output=False):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = input_size or cfg.input_size
        
        if isinstance(cfg.layers, str):
            if cfg.layers.startswith("torchvision:"):
                self.create_layers_from_torchvision_model(cfg.layers.replace("torchvision:", ""))
                layer_configs = []
            else:
                layer_configs = DEFAULT_LAYERS[cfg.layers]
        else:
            layer_configs = cfg.layers

        self.create_layers_from_configs(layer_configs)

        self.output_size_before_flatten = deepcopy(self.output_size)
        if flatten_output and len(self.output_size) > 1:
            self.add_layer(*flatten(None, self.output_size))

        # Optional normalization of encoder output
        self.output_normalizer = get_normalizer(getattr(cfg, "output_normalization", None))

    def create_layers_from_configs(self, layer_configs):
        for layer_cfg in layer_configs:
            self.add_layer(*self.layer_getters[layer_cfg.class_name](layer_cfg, self.output_size))
            if (layer_cfg.activation is not None) and (layer_cfg.class_name != "Activation"):
                self.add_layer(*activation(layer_cfg, self.output_size))

    def add_layer(self, layer, output_size):
        self.layers.append(layer)
        self.output_size = output_size

    def create_layers_from_torchvision_model(self, model_name):
        legal_models = {
            "resnet18": {"out_size": [512, 1, 1]},
        }
        assert model_name in legal_models, f"Invalid torchvision model: {model_name}"
        model = getattr(torchvision_models, model_name)(pretrained=False)
        for layer in list(model.children())[:-1]:
            self.layers.append(layer)

        self.output_size = legal_models[model_name]["out_size"]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_normalizer(x)
        return x


class EncoderList(nn.ModuleList):
    def __init__(self, cfgs=None, input_sizes=None, flatten_output=False, encoder_modules=None):
        super(EncoderList, self).__init__()

        self.output_sizes = []
        self.output_sizes_before_flatten = []

        cfg_none = cfgs is None
        enc_none = encoder_modules is None

        if enc_none and not cfg_none:
            self._init_from_cfgs(cfgs, input_sizes, flatten_output)
        elif cfg_none and not enc_none:
            self._init_from_encoder_modules(encoder_modules)
        else:
            raise RuntimeError("EncoderList expects exactly one of `cfgs` and `encoder_modules` to be not None.")

    def _init_from_cfgs(self, cfgs, input_sizes, flatten_output):
        input_sizes = helpers.ensure_iterable(input_sizes, expected_length=len(cfgs))

        for cfg, input_size in zip(cfgs, input_sizes):
            encoder = Encoder(cfg, input_size=input_size, flatten_output=flatten_output)
            self.append(encoder)
            self.output_sizes.append(encoder.output_size)
            self.output_sizes_before_flatten.append(encoder.output_size_before_flatten)

    def _init_from_encoder_modules(self, encoder_modules):
        for encoder in encoder_modules:
            self.append(encoder)
            self.output_sizes.append(getattr(encoder, "output_size", None))
            self.output_sizes_before_flatten.append(getattr(encoder, "output_size_before_flatten", None))

    def forward(self, views):
        return [e(view) for (e, view) in zip(self, views)]
