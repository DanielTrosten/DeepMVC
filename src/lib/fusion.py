import torch as th
import torch.nn as nn
from copy import deepcopy

import helpers
from lib.encoder import Encoder


class Concat(nn.Module):
    def __init__(self, cfg, input_sizes):
        super(Concat, self).__init__()

        self.check_input_sizes(input_sizes)
        self.output_size = [sum([s[0] for s in input_sizes])]

    @staticmethod
    def check_input_sizes(input_sizes):
        assert all([len(s) == 1 for s in input_sizes]), "Concat-fusion expects all input sizes to have length 1."

    @staticmethod
    def forward(inputs):
        return th.cat(inputs, dim=1)


class WeightedMean(nn.Module):
    def __init__(self, cfg, input_sizes):
        super().__init__()
        self.check_input_sizes(input_sizes)
        self.cfg = cfg
        self.output_size = deepcopy(input_sizes[0])

        n_inputs = len(input_sizes)
        self.weights = nn.Parameter(th.full((n_inputs,), 1 / n_inputs), requires_grad=cfg.trainable_weights)

    @staticmethod
    def check_input_sizes(input_sizes):
        assert all([len(s) == 1 for s in input_sizes]), "Weighted mean fusion expects all input sizes to have length 1."
        assert all([s == input_sizes[0] for s in input_sizes]), "Weighted mean fusion expects all input sizes to be " \
                                                                "equal."

    def forward(self, inputs):
        weights = nn.functional.softmax(self.weights, dim=0)[:, None, None]
        fused = th.sum(weights * th.stack(inputs, dim=0), dim=0)
        return fused


class MLPFusion(nn.Module):
    def __init__(self, cfg, input_sizes):
        super(MLPFusion, self).__init__()

        assert all([len(s) == 1 for s in input_sizes]), f"Invalid input sizes {input_sizes} for MLPFusion."

        self.input_size = [sum([s[0] for s in input_sizes])]
        self.mlp = Encoder(cfg.mlp_config, self.input_size)
        self.output_size = self.mlp.output_size

    def forward(self, inputs):
        inp = th.cat(inputs, dim=1)
        out = self.mlp(inp)
        return out


def get_fusion_module(cfg, input_sizes):
    return helpers.dict_selector({
        "WeightedMean": WeightedMean,
        "Concat": Concat,
        "MLPFusion": MLPFusion,
    }, identifier="fusion module")(cfg.class_name)(cfg, input_sizes)
