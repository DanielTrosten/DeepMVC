from config.experiments.base_experiments import *
from config.templates.models.simvc_comvc import CoMVC


blobs_comvc = BlobsExperiment(
    model_config=CoMVC(
        encoder_configs=BLOBS_ENCODERS
    ),
    batch_size=100,
)

noisymnist_comvc = NoisyMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

edgemnist_comvc = EdgeMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    ),
    batch_size=100,
)

caltech20_comvc = Caltech20Experiment(
    model_config=CoMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

caltech7_comvc = Caltech7Experiment(
    model_config=CoMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_comvc = NoisyFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

edgefashionmnist_comvc = EdgeFashionMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

coil20_comvc = COIL20Experiment(
    model_config=CoMVC(
        encoder_configs=COIL20_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_comvc = PatchedMNISTExperiment(
    model_config=CoMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)
