from .ddc import DDC
from .hp_ddc import HPDDC

import helpers


def get_clustering_module(cfg, input_size):
    return helpers.dict_selector({
        "DDC": DDC,
        "HPDDC": HPDDC,
    }, "clustering module")(cfg.class_name)(cfg, input_size)
