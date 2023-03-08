from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.nn as nn

import helpers
from lib.kernel import cdist


def get_kernel_width_module(cfg, input_size):
    return helpers.dict_selector(dct={
        "Constant": Constant,
        "AMISE": AMISE,
        "MomentumAMISE": MomentumAMISE,
        "MedianDistance": MedianDistance,
        "MomentumMedianDistance": MomentumMedianDistance,
    }, identifier="kernel width")(cfg.class_name)(cfg, input_size)


class _KernelWidth(ABC, nn.Module):
    def __init__(self, cfg, input_size):
        super(_KernelWidth, self).__init__()
        self.cfg = cfg
        self.input_size = input_size
        initial_value = cfg.initial_value or np.nan
        self.register_buffer("kernel_width", tensor=th.tensor(initial_value))

    @abstractmethod
    def forward(self, inputs=None, distances=None, assignments=None):
        pass

    @th.no_grad()
    def update_buffer(self, new):
        self.kernel_width = self.cfg.momentum * self.kernel_width + (1 - self.cfg.momentum) * new


# ======================================================================================================================
# Constant kernel width
# ======================================================================================================================

class Constant(_KernelWidth):
    def __init__(self, cfg, input_size):
        super(Constant, self).__init__(cfg, input_size)

    def forward(self, inputs=None, distances=None, assignments=None):
        return self.kernel_width


# ======================================================================================================================
#  Estimators based on asymptotic mean integrated squared error (AMISE)
# ======================================================================================================================

class AMISE(_KernelWidth):
    def __init__(self, cfg, input_size):
        super(AMISE, self).__init__(cfg, input_size)

        n, p = cfg.batch_size, input_size[0]
        self.factor = (4 / (n * (p + 2))) ** (1 / (p + 4))

        if cfg.std_estimator == "global":
            self.std_estimator = self._global_stds
        elif cfg.std_estimator == "within_cluster":
            self.std_estimator = self._within_cluster_stds
        else:
            raise RuntimeError(f"Invalid estimator for standard deviation: {cfg.std_estimator}.")

    @staticmethod
    def _global_stds(inputs, assignments):
        with th.no_grad():
            return th.mean(th.std(inputs, dim=0))

    @staticmethod
    def _within_cluster_stds(inputs, assignments):
        with th.no_grad():
            freq = assignments.sum(dim=0, keepdims=True)
            centers = (assignments.T @ inputs) / freq.T
            squared_dif = (inputs[:, None, :] - centers[None, :, :]) ** 2
            weighted_squared_dif = assignments[:, :, None] * squared_dif / freq[:, :, None]
            sigma = th.sqrt(weighted_squared_dif.sum(dim=0)).mean()
            return sigma

    def forward(self, inputs=None, distances=None, assignments=None):
        return self.factor * self.std_estimator(inputs, assignments)


class MomentumAMISE(AMISE):
    def forward(self, inputs=None, distances=None, assignments=None):
        if self.training:
            width = self.factor * self.std_estimator(inputs, assignments)
            self.update_buffer(width)
        else:
            width = self.kernel_width
        return width


# ======================================================================================================================
#  Estimators based on median pairwise distances
# ======================================================================================================================

class MedianDistance(_KernelWidth):
    def __init__(self, cfg, input_size):
        super(MedianDistance, self).__init__(cfg, input_size)

    @th.no_grad()
    def _calc_width(self, inputs, distances):
        if distances is None:
            distances = cdist(inputs, inputs)
        width = th.sqrt(self.cfg.rel_sigma * th.median(distances))
        return width

    def forward(self, inputs=None, distances=None, assignments=None):
        return self._calc_width(inputs, distances)


class MomentumMedianDistance(MedianDistance):
    def forward(self, inputs=None, distances=None, assignments=None):
        if self.training:
            width = self._calc_width(inputs, distances)
            self.update_buffer(width)
        else:
            width = self.kernel_width
        return width
