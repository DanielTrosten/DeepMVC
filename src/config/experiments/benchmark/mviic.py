from config.experiments.base_experiments import *
from config.templates.models.mviic import MvIIC

CLUSTERING_HEAD = lambda n_clusters: encoder.Encoder(layers=[layers.Dense(n_units=n_clusters)])
OVERCLUSTERING_HEAD = lambda n_overclusters: encoder.Encoder(layers=[layers.Dense(n_units=n_overclusters)])


blobs_mviic = BlobsExperiment(
    model_config=MvIIC(
        encoder_configs=BLOBS_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(3),
        overclustering_head_config=OVERCLUSTERING_HEAD(10),
    )
)

noisymnist_mviic = NoisyMNISTExperiment(
    model_config=MvIIC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(10),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

edgemnist_mviic = EdgeMNISTExperiment(
    model_config=MvIIC(
        encoder_configs=EDGE_MNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(10),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

caltech20_mviic = Caltech20Experiment(
    model_config=MvIIC(
        encoder_configs=CALTECH_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(20),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

caltech7_mviic = Caltech7Experiment(
    model_config=MvIIC(
        encoder_configs=CALTECH_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(7),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

noisyfashionmnist_mviic = NoisyFashionMNISTExperiment(
    model_config=MvIIC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(10),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

edgefashionmnist_mviic = EdgeFashionMNISTExperiment(
    model_config=MvIIC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(10),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

coil20_mviic = COIL20Experiment(
    model_config=MvIIC(
        encoder_configs=COIL20_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(20),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)

patchedmnist_mviic = PatchedMNISTExperiment(
    model_config=MvIIC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(3),
        overclustering_head_config=OVERCLUSTERING_HEAD(100),
    )
)
