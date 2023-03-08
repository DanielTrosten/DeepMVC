import torch.nn as nn

import helpers
from lib.loss import utils
from register import LOSS_TERM_CLASSES


# Functions to compute the required tensors for the terms.
EXTRA_FUNCS = {
    "hidden_kernel": utils.hidden_kernel,
    "encoder_kernels": utils.encoder_kernels,
    "fused_kernel": utils.fused_kernel,
}


# ======================================================================================================================
# Loss class
# ======================================================================================================================

class Loss(nn.Module):
    def __init__(self, cfg, net):
        """
        Implementation of a general loss function

        :param cfg: Loss function config
        :type cfg: config.defaults.Loss
        """
        super().__init__()
        self.cfg = cfg

        self.names = cfg.funcs.split("|")
        self.weights = cfg.weights if cfg.weights is not None else len(self.names) * [1]

        assert len(self.names) == len(self.weights), f"Mismatch between length of loss-term-names ({self.names}) and " \
                                                     f"length of weights ({self.weights})."

        self.terms = nn.ModuleList()
        for term_name in self.names:
            # We expect term names to be on the form 'loss_name[:param1=value1,param2=value2,...]'
            name = term_name.split(":")
            if len(name) == 1:
                # No params given
                name, params = name[0], None
            elif len(name) == 2:
                # Params given
                name, params = name[0], helpers.parse_str_params(name[1])
            else:
                raise RuntimeError(f"Invalid format for loss term '{term_name}'")
            # Instantiate the loss term
            self.terms.append(LOSS_TERM_CLASSES[name](cfg, params, net))

        self.required_extras_names = list(set(sum([t.required_tensors for t in self.terms], [])))

    def calc_extra(self, net):
        extra = {}
        for name in self.required_extras_names:
            tensors_dict = EXTRA_FUNCS[name](net, self.cfg)
            extra.update(tensors_dict)
        return extra

    def forward(self, net, ignore_in_total=tuple()):
        extra = self.calc_extra(net)
        loss_values = {}
        for name, term, weight in zip(self.names, self.terms, self.weights):
            value = term(net, self.cfg, extra)
            # If we got a dict, add each term from the dict with "name/" as the scope.
            if isinstance(value, dict):
                for key, _value in value.items():
                    loss_values[f"{name}/{key}"] = weight * _value
            # Otherwise, just add the value to the dict directly
            else:
                loss_values[name] = weight * value

        loss_values["tot"] = sum([loss_values[k] for k in loss_values.keys() if k not in ignore_in_total])
        return loss_values
