from config.experiments.base_experiments import *
from config.templates.models.dmsc import DMSC

DATASET_KWARGS = dict(
    pre_train_batch_size=100,
    pre_train_train_shuffle=True,
    pre_train_val_shuffle=True,
    train_shuffle=False,
    val_shuffle=False,
    test_shuffle=False,
)


blobs_dmsc = BlobsExperiment(
    dataset_config=Dataset(name="blobs_dep", **DATASET_KWARGS),
    model_config=DMSC(
        n_samples=3000,
        encoder_configs=BLOBS_ENCODERS,
        decoder_configs=BLOBS_DECODERS,
    ),
    batch_size=3000,
)


noisymnist_dmsc = NoisyMNISTExperiment(
    dataset_config=Dataset(name="noisymnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

edgemnist_dmsc = EdgeMNISTExperiment(
    dataset_config=Dataset(name="edgemnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

noisyfashionmnist_dmsc = NoisyFashionMNISTExperiment(
    dataset_config=Dataset(name="noisyfashionmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

edgefashionmnist_dmsc = EdgeFashionMNISTExperiment(
    dataset_config=Dataset(name="edgefashionmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)

caltech20_dmsc = Caltech20Experiment(
    dataset_config=Dataset(name="caltech20", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        n_samples=2386,
    ),
    batch_size=2386
)

caltech7_dmsc = Caltech7Experiment(
    dataset_config=Dataset(name="caltech7", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        n_samples=1474,
    ),
    batch_size=1474
)

coil20_dmsc = COIL20Experiment(
    dataset_config=Dataset(name="coil20", **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        n_samples=480,
    ),
    batch_size=480
)

patchedmnist_dmsc = COIL20Experiment(
    dataset_config=Dataset(name="patchedmnist", n_train_samples=3000, **DATASET_KWARGS),
    model_config=DMSC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        n_samples=3000,
    ),
    batch_size=3000
)
