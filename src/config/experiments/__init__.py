# Benchmark experiments
from .benchmark.simvc import *
from .benchmark.comvc import *
from .benchmark.eamc import *
from .benchmark.mvae import *
from .benchmark.mvscn import *
from .benchmark.dmsc import *
from .benchmark.contrastive_ae import *
from .benchmark.mimvc import *
from .benchmark.mviic import *

# Ablation study experiments
from .ablation.sv_ssl import *
from .ablation.mv_ssl import *
from .ablation.fusion import *

# Increasing views experiments
from config.experiments.increasing_n_views.caltech7 import *
from config.experiments.increasing_n_views.patchedmnist import *
