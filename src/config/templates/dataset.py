from typing import Tuple, List, Optional

from config import Config, constants
from config.templates import models, augmenter


class Dataset(Config):
    # Name of the dataset. Must correspond to a filename in data/processed/
    name: str

    # Seed for random data loading ops.
    random_seed: int = 7

    # Include indices of batch elements in the dataset
    include_index: bool = False

    # Number of samples to load. Set to None to load all samples
    n_train_samples: int = None
    n_val_samples: int = None
    n_test_samples: int = None

    # Subset of views to load. Set to None to load all views
    select_views: List[int] = None

    # Subset of labels (classes) to load. Set to None to load all classes
    select_labels: List[int] = None

    # Number of samples to load for each class. Set to None to load all samples
    train_label_counts: List[int] = None
    val_label_counts: List[int] = None
    test_label_counts: List[int] = None

    # Drop last batch (if not a complete batch), when dataset is batched.
    drop_last: bool = True

    # Whether to shuffle the validation and test data
    train_shuffle: bool = True
    val_shuffle: bool = True
    test_shuffle: bool = False

    # Number of DataLoader workers
    n_train_workers: int = 0
    n_val_workers: int = 0
    n_test_workers: int = 0

    # Prefetch factor for train dataloader (only used when n_train_workers > 0).
    prefetch_factor = 1

    # Config for data augmentation. Set to None to disable augmentation.
    augmenter_configs: List[augmenter.Augmenter] = None

    # Pre-train-specific parameters. Set to None to use same values as in fine-tune (specified above)
    pre_train_batch_size: int = None
    pre_train_train_shuffle: bool = None
    pre_train_val_shuffle: bool = None
    pre_train_test_shuffle: bool = None

    # Batch size (This is a placeholder. Set the batch size in Experiment).
    batch_size: int = None
