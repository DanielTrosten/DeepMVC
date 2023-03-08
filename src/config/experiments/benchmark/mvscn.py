from config.experiments.base_experiments import *
from config.templates import encoder, layers
from config.templates.models.mvscn import SiameseNet, MvSCN


def default_head(n_clusters, activation="tanh"):
    return encoder.Encoder(layers=[layers.Dense(n_units=n_clusters, activation=activation)])


blobs_siam = BlobsExperiment(
    dataset_config=Dataset(name="blobs_dep_paired"),
    model_config=SiameseNet(
        encoder_configs=BLOBS_ENCODERS,
        head_configs=default_head(3, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
)

blobs_mvscn = BlobsExperiment(
    model_config=MvSCN(
        encoder_configs=BLOBS_ENCODERS,
        head_configs=default_head(3),
        siam_dir="blobs_siam-11pid8ct/run-0",
    ),
)

noisymnist_siam = NoisyMNISTExperiment(
    dataset_config=Dataset(name="noisymnist_paired", n_train_samples=70000),
    model_config=SiameseNet(
        encoder_configs=NOISY_MNIST_ENCODERS,
        head_configs=default_head(10, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

noisymnist_mvscn = NoisyMNISTExperiment(
    model_config=MvSCN(
        encoder_configs=NOISY_MNIST_ENCODERS,
        head_configs=default_head(10),
        siam_dir="noisymnist_siam-28j69fly/run-0",
    ),
)

noisyfashionmnist_siam = NoisyFashionMNISTExperiment(
    dataset_config=Dataset(name="noisyfashionmnist_paired", n_train_samples=70000),
    model_config=SiameseNet(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        head_configs=default_head(10, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

noisyfashionmnist_mvscn = NoisyFashionMNISTExperiment(
    model_config=MvSCN(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        head_configs=default_head(10),
        siam_dir="noisyfashionmnist_siam-6sb6hraj/run-0",
    ),
)

edgemnist_siam = EdgeMNISTExperiment(
    dataset_config=Dataset(name="edgemnist_paired", n_train_samples=70000),
    model_config=SiameseNet(
        encoder_configs=EDGE_MNIST_ENCODERS,
        head_configs=default_head(10, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

edgemnist_mvscn = EdgeMNISTExperiment(
    model_config=MvSCN(
        encoder_configs=EDGE_MNIST_ENCODERS,
        head_configs=default_head(10),
        siam_dir="edgemnist_siam-33cz0b6a/run-0",
    ),
)

edgefashionmnist_siam = EdgeFashionMNISTExperiment(
    dataset_config=Dataset(name="edgefashionmnist_paired", n_train_samples=70000),
    model_config=SiameseNet(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        head_configs=default_head(10, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

edgefashionmnist_mvscn = EdgeFashionMNISTExperiment(
    model_config=MvSCN(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        head_configs=default_head(10),
        siam_dir="edgefashionmnist_siam-2n5lc4cw/run-0",
    ),
)

coil20_siam = COIL20Experiment(
    dataset_config=Dataset(name="coil20_paired", n_train_samples=480),
    model_config=SiameseNet(
        encoder_configs=COIL20_ENCODERS,
        head_configs=default_head(20, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

coil20_mvscn = COIL20Experiment(
    model_config=MvSCN(
        encoder_configs=COIL20_ENCODERS,
        head_configs=default_head(20),
        siam_dir="coil20_siam-1fpaugiq/run-0",
    ),
)

caltech20_siam = Caltech20Experiment(
    dataset_config=Dataset(name="caltech20_paired", n_train_samples=2386),
    model_config=SiameseNet(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=default_head(20, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

caltech20_mvscn = Caltech20Experiment(
    model_config=MvSCN(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=default_head(20),
        siam_dir="caltech20_siam-1rtcgwq0/run-0",
    ),
)

caltech7_siam = Caltech7Experiment(
    dataset_config=Dataset(name="caltech7_paired", n_train_samples=1474),
    model_config=SiameseNet(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=default_head(7, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

caltech7_mvscn = Caltech7Experiment(
    model_config=MvSCN(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=default_head(7),
        siam_dir="caltech7_siam-2vrj0cvd/run-0",
    ),
)

patchedmnist_siam = PatchedMNISTExperiment(
    dataset_config=Dataset(name="patchedmnist_paired", n_train_samples=21770),
    model_config=SiameseNet(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        head_configs=default_head(3, activation="relu"),
    ),
    n_runs=1,
    batch_size=128,
    wandb_tags="hidden"
)

patchedmnist_mvscn = PatchedMNISTExperiment(
    model_config=MvSCN(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        head_configs=default_head(3),
        siam_dir="patchedmnist_siam-/run-0",
    ),
)

