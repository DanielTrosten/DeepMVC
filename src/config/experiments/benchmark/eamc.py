from config.experiments.base_experiments import *
from config.templates.models.eamc import EAMC


blobs_eamc = BlobsExperiment(
    model_config=EAMC(
        encoder_configs=BLOBS_ENCODERS,
    ),
    batch_size=100,
)

noisymnist_eamc = NoisyMNISTExperiment(
    model_config=EAMC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

edgemnist_eamc = EdgeMNISTExperiment(
    model_config=EAMC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    ),
    batch_size=100,
)

caltech20_eamc = Caltech20Experiment(
    model_config=EAMC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

caltech7_eamc = Caltech7Experiment(
    model_config=EAMC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_eamc = NoisyFashionMNISTExperiment(
    model_config=EAMC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

edgefashionmnist_eamc = EdgeFashionMNISTExperiment(
    model_config=EAMC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

coil20_eamc = COIL20Experiment(
    model_config=EAMC(
        encoder_configs=COIL20_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_eamc = PatchedMNISTExperiment(
    model_config=EAMC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)