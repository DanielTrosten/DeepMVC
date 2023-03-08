from typing import Tuple, List, Optional

from config import Config, constants
from config.templates.dataset import Dataset


class Experiment(Config):
    _glob_vars: Tuple[str, ...] = ("n_clusters", "batch_size", "n_views")
    tied_args: str = None

    # Dataset config
    dataset_config: Dataset

    # Number of clusters
    n_clusters: int
    # Number of views
    n_views: int
    # Batch size
    batch_size: int = 100

    # Model config
    model_config: Config
    # Number of training runs
    n_runs: int = 5
    # Number of training epochs
    n_epochs: int = 100
    # Number of pre-training epochs (only used when pre-training is enabled in model_config).
    n_pre_train_epochs = 50
    # Number of epochs between model evaluation.
    eval_interval: int = 4
    # Number of epochs between model checkpoints.
    checkpoint_interval: int = 50
    # Patience for early stopping.
    patience: int = 50000
    # Number of sanity-check iterations
    num_sanity_val_steps: int = 2
    # Number of samples to use for evaluation. Set to None to use all samples in the dataset.
    n_eval_samples: int = None
    # Term in loss function to use for model selection. Set to "tot" to use the sum of all terms.
    best_loss_term: str = "tot"
    is_sweep: bool = False
    gpus: int = constants.GPUS
    detect_anomaly: bool = False
    wandb_tags: str = None

    # Determinism stuff
    trainer_deterministic: bool = True
    everything_seed: Optional[int] = None
