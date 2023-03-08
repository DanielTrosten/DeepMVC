from config.experiments.base_experiments import *
from config.templates.models.simvc_comvc import SiMVC


noisymnist_simvc = NoisyMNISTExperiment(
    model_config=SiMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
    ),
    batch_size=100,
)

edgemnist_simvc = EdgeMNISTExperiment(
    model_config=SiMVC(
        encoder_configs=EDGE_MNIST_ENCODERS,
    ),
    batch_size=100,
)

caltech20_simvc = Caltech20Experiment(
    model_config=SiMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

caltech7_simvc = Caltech7Experiment(
    model_config=SiMVC(
        encoder_configs=CALTECH_ENCODERS,
    ),
    batch_size=100,
)

noisyfashionmnist_simvc = NoisyFashionMNISTExperiment(
    model_config=SiMVC(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

edgefashionmnist_simvc = EdgeFashionMNISTExperiment(
    model_config=SiMVC(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
    ),
    batch_size=100,
)

coil20_simvc = COIL20Experiment(
    model_config=SiMVC(
        encoder_configs=COIL20_ENCODERS,
    ),
    batch_size=100,
)

patchedmnist_simvc = PatchedMNISTExperiment(
    model_config=SiMVC(
        encoder_configs=PATCHED_MNIST_ENCODERS,
    ),
    batch_size=100,
)