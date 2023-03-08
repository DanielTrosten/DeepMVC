from config import Config
from config.templates import kernel_width


class ClusteringModule(Config):
    # Number of clusters (Placeholder)
    n_clusters: int = None

    kernel_width_config: kernel_width.KernelWidth = kernel_width.MedianDistance(
        rel_sigma=0.15
    )


class DDC(ClusteringModule):
    # Number of clusters
    n_clusters: int = None
    # Number of units in the first fully connected layer
    n_hidden: int = 100
    # Use batch norm after the first fully connected layer?
    use_bn: bool = True
    bn_trainable_params: bool = True
