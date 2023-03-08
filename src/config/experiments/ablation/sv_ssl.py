from config.experiments.base_experiments import *
from config.templates.models import mvscn, custom, dmsc


ablsvssl_noisymnist_mvscn = NoisyMNISTExperiment(
    model_config=mvscn.MvSCN(
        encoder_configs=NOISY_MNIST_ENCODERS,
        head_configs=encoder.Encoder(layers=[layers.Dense(n_units=10)]),
        loss_config=mvscn.MVSCNLoss(
            funcs="MVSCN2",
            weights=[1.0],
        ),
        siam_dir="noisymnist_siam-28j69fly/run-0",
    ),
    wandb_tags="ablation,sv-ssl"
)


ablsvssl_noisymnist_caekm = NoisyMNISTExperiment(
    model_config=custom.CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=custom.CAEKMLoss(
            funcs="contrast",
            weights=[1.0],
        )
    ),
    wandb_tags="ablation,sv-ssl"
)

ablsvssl_noisymnist_saekm = NoisyMNISTExperiment(
    model_config=custom.CAEKM(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=custom.CAEKMLoss(
            funcs="Zero",
            weights=None,
        )
    ),
    wandb_tags="ablation,sv-ssl",
    # Train for zero epochs since we have no loss.
    n_epochs=0,
)

ablsvssl_caltech7_mvscn = Caltech7Experiment(
    model_config=mvscn.MvSCN(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=encoder.Encoder(layers=[layers.Dense(n_units=10)]),
        loss_config=mvscn.MVSCNLoss(
            funcs="MVSCN2",
            weights=[1.0],
        ),
        siam_dir="caltech7_siam-2vrj0cvd/run-0",
    ),
    wandb_tags="ablation,sv-ssl"
)


ablsvssl_caltech7_caekm = Caltech7Experiment(
    model_config=custom.CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=custom.CAEKMLoss(
            funcs="contrast",
            weights=[1.0],
        )
    ),
    wandb_tags="ablation,sv-ssl"
)

ablsvssl_caltech7_saekm = Caltech7Experiment(
    model_config=custom.CAEKM(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=custom.CAEKMLoss(
            funcs="Zero",
            weights=None,
        )
    ),
    wandb_tags="ablation,sv-ssl",
    n_epochs=0,
)


# ======================================================================================================================
# DMSC
# ======================================================================================================================

DMSC_DATASET_KWARGS = dict(
    pre_train_batch_size=100,
    pre_train_train_shuffle=True,
    pre_train_val_shuffle=True,
    train_shuffle=False,
    val_shuffle=False,
    test_shuffle=False,
)


ablsvssl_noisymnist_dmsc = NoisyMNISTExperiment(
    dataset_config=Dataset(name="noisymnist", n_train_samples=3000, **DMSC_DATASET_KWARGS),
    model_config=dmsc.DMSC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        n_samples=3000,
        pre_train_loss_config=None,
        loss_config=dmsc.DMSCLoss(
            funcs="DMSC1|DMSC2",
            weights=None,
        )
    ),
    batch_size=3000
)


ablsvssl_caltech7_dmsc = Caltech7Experiment(
    dataset_config=Dataset(name="caltech7", **DMSC_DATASET_KWARGS),
    model_config=dmsc.DMSC(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        n_samples=1474,
        pre_train_loss_config=None,
        loss_config=dmsc.DMSCLoss(
            funcs="DMSC1|DMSC2",
            weights=None,
        )
    ),
    batch_size=1474
)
