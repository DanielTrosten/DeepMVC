from config.experiments.base_experiments import *
from config.templates.models.mvae import MultiVAE


blobs_mvae = BlobsExperiment(
    model_config=MultiVAE(
        encoder_configs=BLOBS_ENCODERS,
        decoder_configs=BLOBS_DECODERS,
        hidden_dim=8,
        cont_dim=8,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

noisymnist_mvae = NoisyMNISTExperiment(
    model_config=MultiVAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

edgemnist_mvae = EdgeMNISTExperiment(
    model_config=MultiVAE(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

caltech20_mvae = Caltech20Experiment(
    model_config=MultiVAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

caltech7_mvae = Caltech7Experiment(
    model_config=MultiVAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

noisyfashionmnist_mvae = NoisyFashionMNISTExperiment(
    model_config=MultiVAE(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

edgefashionmnist_mvae = EdgeFashionMNISTExperiment(
    model_config=MultiVAE(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

coil20_mvae = COIL20Experiment(
    model_config=MultiVAE(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)

patchedmnist_mvae = PatchedMNISTExperiment(
    model_config=MultiVAE(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
    ),
    batch_size=64,
    num_sanity_val_steps=0,
)