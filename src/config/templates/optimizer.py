from typing import List, Union, Optional
from typing_extensions import Literal

from config import Config
from config.templates import kernel_width, fusion
from config.templates.encoder import Encoder


class Scheduler(Config):
    # Step size for the learning rate scheduler. None disables the scheduler.
    step_size: int = 50
    # Multiplication factor for the learning rate scheduler
    gamma: float = 0.1
    # Number of epochs to use in linear warm-up. Set to None to disable
    warmup_epochs: int = None


class Optimizer(Config):
    opt_type: Literal["adam", "sgd"] = "adam"
    # Base learning rate
    learning_rate: float = 1e-3
    # SGD momentum
    sgd_momentum: float = 0.0
    # Max gradient norm for gradient clipping.
    clip_norm: Optional[float] = 10.0
    # Config for the (optional) LR scheduler.
    scheduler_config: Scheduler = None
