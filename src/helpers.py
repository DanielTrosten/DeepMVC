import math
import faiss
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn

import config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def npy(t, to_cpu=True):
    """
    Convert a tensor to a numpy array.

    :param t: Input tensor
    :type t: th.Tensor
    :param to_cpu: Call the .cpu() method on `t`?
    :type to_cpu: bool
    :return: Numpy array
    :rtype: np.ndarray
    """
    if isinstance(t, (list, tuple)):
        # We got a list. Convert each element to numpy
        return [npy(ti) for ti in t]
    elif isinstance(t, dict):
        # We got a dict. Convert each value to numpy
        return {k: npy(v) for k, v in t.items()}
    # Assuming t is a tensor.
    if to_cpu:
        return t.cpu().detach().numpy()
    return t.detach().numpy()


def ensure_iterable(elem, expected_length=1, assert_length=True):
    if isinstance(elem, (list, tuple)):
        if assert_length:
            assert len(elem) == expected_length, f"Expected iterable {elem} with length {len(elem)} does not have " \
                                                 f"expected length {expected_length}"
    else:
        elem = expected_length * [elem]
    return elem


def dict_means(dicts):
    """
    Compute the mean value of keys in a list of dicts

    :param dicts: Input dicts
    :type dicts: List[dict]
    :return: Mean values
    :rtype: dict
    """
    return pd.DataFrame(dicts).mean(axis=0).to_dict()


def add_prefix(dct, prefix, sep="/", inplace=False):
    """
    Add a prefix to all keys in `dct`.

    :param dct: Input dict
    :type dct: dict
    :param prefix: Prefix
    :type prefix: str
    :param sep: Separator between prefix and key
    :type sep: str
    :return: Dict with prefix prepended to all keys
    :rtype: dict
    """
    if not inplace:
        return {prefix + sep + key: value for key, value in dct.items()}

    keys = list(dct.keys())
    for key in keys:
        dct[prefix + sep + key] = dct[key]

    for k in keys:
        del dct[k]


def move_elem_to_idx(lst, elem, idx, twins=tuple()):
    current_idx = lst.index(elem)
    lst.insert(idx, lst.pop(current_idx))
    for twin in twins:
        twin.insert(idx, twin.pop(current_idx))


def recursive_to(it, **kwargs):
    if isinstance(it, (list, tuple)):
        return type(it)([recursive_to(elem, **kwargs) for elem in it])
    return it.to(**kwargs)


def get_save_dir(experiment_name, identifier, run):
    """
    Get the save dir for an experiment

    :param experiment_name: Name of the config
    :type experiment_name: str
    :param identifier: 8-character unique identifier for the current experiment
    :type identifier: str
    :param run: Current training run
    :type run: int
    :return: Path to save dir
    :rtype: pathlib.Path
    """
    base_experiment_dir = config.MODELS_DIR / f"{experiment_name}-{identifier}"
    if run is None:
        return base_experiment_dir

    if isinstance(run, int):
        run = f"run-{run}"
    return base_experiment_dir / run


def he_init_weights(module):
    """
    Initialize network weights using the He (Kaiming) initialization strategy.

    :param module: Network module
    :type module: nn.Module
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight)


def parse_str_params(s):
    params = {}
    for param in s.split(","):
        key, value = param.split("=")
        params[key] = value
    return params


def num2tuple(num):
    return num if isinstance(num, (tuple, list)) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Compute the output shape of a convolution operation.

    :param h_w: Height and width of input
    :type h_w: Tuple[int, int]
    :param kernel_size: Size of kernel
    :type kernel_size: Union[int, Tuple[int, int]]
    :param stride: Stride of convolution
    :type stride: Union[int, Tuple[int, int]]
    :param pad: Padding (in pixels)
    :type pad: Union[int, Tuple[int, int]]
    :param dilation: Dilation
    :type dilation: Union[int, Tuple[int, int]]
    :return: Height and width of output
    :rtype: Tuple[int, int]
    """
    h_w, kernel_size, stride, = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride)
    pad, dilation = num2tuple(pad), num2tuple(dilation)

    h = math.floor((h_w[0] + 2 * pad[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + 2 * pad[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return h, w


def conv_transpose_2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, output_pad=0):
    h_w, kernel_size, stride, = num2tuple(h_w), num2tuple(kernel_size), num2tuple(stride)
    pad, dilation, output_pad = num2tuple(pad), num2tuple(dilation), num2tuple(output_pad)
    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + dilation[0] * (kernel_size[0] - 1) + output_pad[0] + 1
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + dilation[1] * (kernel_size[1] - 1) + output_pad[1] + 1
    return h, w


def conv1d_output_shape(l_old, padding=0, dilation=1, kernel_size=1, stride=1, **_):
    l_new = math.floor((l_old + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return l_new


def dict_selector(dct, identifier="", exception_class=RuntimeError):
    def func(key):
        try:
            return dct[key]
        except KeyError as err:
            raise exception_class(f"Invalid {identifier}: '{key}'") from err
    return func


def faiss_kmeans(x, k, n_iter=20, n_redo=10, spherical=False, verbose=False, gpu=None):
    x = np.ascontiguousarray(x)
    kmeans = faiss.Kmeans(
        x.shape[1], k,
        niter=n_iter,
        nredo=n_redo,
        spherical=spherical,
        verbose=verbose,
        gpu=(gpu if gpu is not None else config.CUDA_AVALABLE),
    )
    kmeans.train(x)
    centroids = kmeans.centroids
    assignments = kmeans.index.search(x, 1)[1].squeeze()
    obj = np.max(kmeans.obj) if spherical else np.min(kmeans.obj)
    return assignments, centroids, obj
