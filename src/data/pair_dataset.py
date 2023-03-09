import faiss
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm

import config

PROCESSED_DIR = config.DATA_DIR / "processed"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, dest="dataset", type=str)
    parser.add_argument("-k", dest="k", default=3, type=int)
    return parser.parse_args()


def make_index(x, gpu=False):
    index = faiss.IndexFlatL2(x.shape[1])
    if gpu:
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            index,
        )
    index.add(x)
    return index


def generate_pairs(x, k):
    x = x.astype(np.float32)
    x_flat = np.ascontiguousarray(x.reshape((x.shape[0], -1)))

    index = make_index(x_flat)
    ngh_idx = index.search(x_flat, k + 1)[0][:, 1:].astype(int)
    rng = np.random.default_rng(7)
    n_samples = x.shape[0]
    rang = np.arange(n_samples)

    pairs, labels = [], []

    for i in tqdm(range(n_samples)):
        pos = [np.stack((x[i], x[j]), axis=0) for j in ngh_idx[i]]
        pairs += pos

        neg_weights = np.ones(n_samples)
        neg_weights[ngh_idx[i]] = 0
        neg_weights[i] = 0
        neg_weights /= neg_weights.sum()
        neg_idx = rng.choice(rang, size=k, replace=False, p=neg_weights)

        neg = [np.stack((x[i], x[j]), axis=0) for j in neg_idx]
        pairs += neg

        labels += (k * [1]) + (k * [0])

    pairs = np.stack(pairs, axis=0)
    labels = np.array(labels)

    print("Paired shape:", pairs.shape)
    print("Pair labels shape:", labels.shape)
    assert labels.shape[0] == pairs.shape[0]
    return pairs, labels


def main():
    args = parse_args()
    data = np.load(str(PROCESSED_DIR / f"{args.dataset}_train.npz"))
    n_views = data["n_views"]
    views = [data[f"view_{v}"] for v in range(n_views)]

    new_labels = np.stack(2 * args.k * [data["labels"]], axis=1).ravel()
    pair_data = {"n_views": 2 * n_views, "labels": new_labels}
    for i, v in enumerate(views):
        pairs, pair_lab = generate_pairs(x=v, k=args.k)
        pair_data[f"view_{i}"] = pairs
        pair_data[f"view_{n_views + i}"] = pair_lab

    out_file = str(PROCESSED_DIR / f"{args.dataset}_paired_train")
    np.savez(out_file, **pair_data)
    print(f"Successfully saved paired data to '{out_file}'")


if __name__ == '__main__':
    main()
