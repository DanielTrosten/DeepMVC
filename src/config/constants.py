import os
import torch as th
from pathlib import Path


CUDA_AVALABLE = th.cuda.is_available()
GPUS = 1 if CUDA_AVALABLE else 0
DEVICE = th.device("cuda" if CUDA_AVALABLE else "cpu")

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATETIME_FMT = "%Y-%m-%d_%H-%M-%S"

EPSILON = 1E-9
