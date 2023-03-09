import os
import numpy as np
import torch as th
from sklearn import preprocessing, decomposition

import config


def _load_npz(name, split):
    file_path = config.DATA_DIR / "processed" / f"{name}_{split}.npz"
    if not os.path.exists(file_path):
        print(f"Could not find dataset '{name} ({split})' at {file_path}.")
        return None
    return np.load(file_path)


def _fix_labels(l):
    uniq = np.unique(l)[None, :]
    new = (l[:, None] == uniq).argmax(axis=1)
    return new


def _normalize(views, mode):
    if mode == "l2":
        views = [preprocessing.normalize(v, norm="l2") for v in views]
    elif mode == "minmax":
        views = [preprocessing.minmax_scale(v, feature_range=(0, 1)) for v in views]
    else:
        raise RuntimeError(f"Invalid normalization mode: {mode}")
    return views


def _pca(X, out_dim):
    if out_dim is None:
        return X
    return decomposition.PCA(n_components=out_dim).fit_transform(X)


def _flatten_list(lst):
    out = []
    for elem in lst:
        if isinstance(elem, list):
            out += elem
        else:
            out.append(elem)
    return out


def load_dataset(name, split="train", random_seed=None, n_samples=None,
                 select_views=None, select_labels=None, label_counts=None, noise_sd=None, noise_views=None,
                 to_dataset=True, normalization=None, pca_dims=None, include_index=False):

    npz = _load_npz(name, split)
    if npz is None:
        return

    labels = npz["labels"]
    views = [npz[f"view_{i}"] for i in range(npz["n_views"])]

    if random_seed is not None:
        prev_state = np.random.get_state()
        np.random.seed(random_seed)

    if select_labels is not None:
        mask = np.isin(labels, select_labels)
        labels = labels[mask]
        views = [v[mask] for v in views]
        labels = _fix_labels(labels)

    if label_counts is not None:
        idx = []
        unique_labels = np.unique(labels)
        assert len(unique_labels) == len(label_counts)
        for l, n in zip(unique_labels, label_counts):
            _idx = np.random.choice(np.where(labels == l)[0], size=n, replace=False)
            idx.append(_idx)

        idx = np.concatenate(idx, axis=0)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if n_samples is not None:
        idx = np.random.choice(labels.shape[0], size=min(labels.shape[0], int(n_samples)), replace=False)
        labels = labels[idx]
        views = [v[idx] for v in views]

    if select_views is not None:
        if not isinstance(select_views, (list, tuple)):
            select_views = [select_views]
        views = [views[i] for i in select_views]

    if noise_sd is not None:
        assert noise_views is not None, "'noise_views' has to be specified when 'noise_sd' is not None."
        for v in noise_views:
            views[v] += np.random.normal(loc=0, scale=noise_sd, size=views[v].shape)

    if normalization is not None:
        views = _normalize(views, normalization)

    if pca_dims is not None:
        assert len(pca_dims) == len(views)
        views = [_pca(v, d) for v, d in zip(views, pca_dims)]

    views = [v.astype(np.float32) for v in views]

    dataset = [views, labels]

    if include_index:
        dataset.insert(2, np.arange(labels.shape[0]))

    if to_dataset:
        tensors = [th.from_numpy(arr) for arr in _flatten_list(dataset)]
        dataset = th.utils.data.TensorDataset(*tensors)

    if random_seed is not None:
        np.random.set_state(prev_state)

    return dataset
