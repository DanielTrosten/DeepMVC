from config.experiments.base_experiments import *
from config.templates.models.custom import CAE, CAEKM, CAEKMLoss, CAELoss


# ======================================================================================================================
# CAE
# ======================================================================================================================

noisymnist_cae = NoisyMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
    )
)

edgemnist_cae = EdgeMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
    )
)

caltech20_cae = Caltech20Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

caltech7_cae = Caltech7Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

noisyfashionmnist_cae = NoisyFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
    )
)

edgefashionmnist_cae = EdgeFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
    )
)

coil20_cae = COIL20Experiment(
    model_config=CAE(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
    )
)

patchedmnist_cae = PatchedMNISTExperiment(
    model_config=CAE(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
    )
)


# ======================================================================================================================
# CAEKM
# ======================================================================================================================

noisymnist_caekm = NoisyMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
    )
)

edgemnist_caekm = EdgeMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
    )
)

caltech20_caekm = Caltech20Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

caltech7_caekm = Caltech7Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
    )
)

noisyfashionmnist_caekm = NoisyFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
    )
)

edgefashionmnist_caekm = EdgeFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
    )
)

coil20_caekm = COIL20Experiment(
    model_config=CAEKM(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
    )
)

patchedmnist_caekm = PatchedMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
    )
)


# ======================================================================================================================
# SAE
# ======================================================================================================================

noisymnist_sae = NoisyMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgemnist_sae = EdgeMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech20_sae = Caltech20Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech7_sae = Caltech7Experiment(
    model_config=CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

noisyfashionmnist_sae = NoisyFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgefashionmnist_sae = EdgeFashionMNISTExperiment(
    model_config=CAE(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

coil20_sae = COIL20Experiment(
    model_config=CAE(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        loss_config=CAELoss(
            funcs="DDC1|DDC2|DDC3|MSE",
            weights=[1, 1, 1, 0.1],
        ),
        pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

patchedmnist_sae = PatchedMNISTExperiment(
    model_config=CAE(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        loss_config=CAELoss(funcs="DDC1|DDC2|DDC3|MSE"),
        pre_train_loss_config=CAELoss(funcs="MSE"),
        projector_config=None,
    )
)


# ======================================================================================================================
# SAEKM
# ======================================================================================================================

noisymnist_saekm = NoisyMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgemnist_saekm = EdgeMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_MNIST_ENCODERS,
        decoder_configs=EDGE_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech20_saekm = Caltech20Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

caltech7_saekm = Caltech7Experiment(
    model_config=CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

noisyfashionmnist_saekm = NoisyFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=NOISY_FASHIONMNIST_ENCODERS,
        decoder_configs=NOISY_FASHIONMNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

edgefashionmnist_saekm = EdgeFashionMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=EDGE_FASHIONMNIST_ENCODERS,
        decoder_configs=EDGE_FASHIONMNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

coil20_saekm = COIL20Experiment(
    model_config=CAEKM(
        encoder_configs=COIL20_ENCODERS,
        decoder_configs=COIL20_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE", weights=None),
        projector_config=None,
    )
)

patchedmnist_saekm = PatchedMNISTExperiment(
    model_config=CAEKM(
        encoder_configs=PATCHED_MNIST_ENCODERS,
        decoder_configs=PATCHED_MNIST_DECODERS,
        loss_config=CAEKMLoss(funcs="MSE"),
        projector_config=None,
    )
)

