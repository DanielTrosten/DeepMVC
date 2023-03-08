from config.experiments.base_experiments import *
from config.templates.models import eamc, mvscn, simvc_comvc, mvae, mviic
from config.templates.fusion import WeightedMean


# ======================================================================================================================
# EAMC
#
# EAMC w/o MV-SSL corresponds to SiMVC with fixed fusion weights, and DDC2Flipped instead of DDC2
# ======================================================================================================================

ablmvssl_noisymnist_eamc = NoisyMNISTExperiment(
    model_config=simvc_comvc.SiMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        fusion_config=WeightedMean(trainable_weights=False),
        loss_config=eamc.EAMCLoss(
            funcs="DDC1|DDC2Flipped|DDC3",
            weights=[1.0, 1.0, 1.0]
        )
    ),
    wandb_tags="ablation,mv-ssl"
)

ablmvssl_caltech7_eamc = Caltech7Experiment(
    model_config=simvc_comvc.SiMVC(
        encoder_configs=CALTECH_ENCODERS,
        fusion_config=WeightedMean(trainable_weights=False),
        loss_config=eamc.EAMCLoss(
            funcs="DDC1|DDC2Flipped|DDC3",
            weights=[1.0, 1.0, 1.0]
        )
    ),
    wandb_tags="ablation,mv-ssl"
)


# ======================================================================================================================
# MVSCN
# ======================================================================================================================

ablmvssl_noisymnist_mvscn = NoisyMNISTExperiment(
    model_config=mvscn.MvSCN(
        encoder_configs=NOISY_MNIST_ENCODERS,
        head_configs=encoder.Encoder(layers=[layers.Dense(n_units=10)]),
        loss_config=mvscn.MVSCNLoss(
            funcs="MVSCN1",
            weights=[1.0],
        ),
        siam_dir="noisymnist_siam-28j69fly/run-0",
    ),
    wandb_tags="ablation,mv-ssl"
)

ablmvssl_caltech7_mvscn = Caltech7Experiment(
    model_config=mvscn.MvSCN(
        encoder_configs=CALTECH_ENCODERS,
        head_configs=encoder.Encoder(layers=[layers.Dense(n_units=10)]),
        loss_config=mvscn.MVSCNLoss(
            funcs="MVSCN1",
            weights=[1.0],
        ),
        siam_dir="caltech7_siam-2vrj0cvd/run-0",
    ),
    wandb_tags="ablation,mv-ssl"
)


# ======================================================================================================================
# Multi-VAE
#
# Multi-VAE without MV-SSL is essentially a randomly initialized network: Not trained since all losses are MV-SSL
# losses.
# ======================================================================================================================

ablmvssl_noisymnist_mvae = NoisyMNISTExperiment(
    model_config=mvae.MultiVAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=mvae.MultiVAELoss(
            funcs="Zero",
            weights=None
        )
    ),
    batch_size=64,
    num_sanity_val_steps=0,
    wandb_tags="ablation,mv-ssl",
    n_epochs=0,
)

ablmvssl_caltech7_mvae = Caltech7Experiment(
    model_config=mvae.MultiVAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=mvae.MultiVAELoss(
            funcs="Zero",
            weights=None
        )
    ),
    batch_size=64,
    num_sanity_val_steps=0,
    wandb_tags="ablation,mv-ssl",
    n_epochs=0,
)


# ======================================================================================================================
# MvIIC
# ======================================================================================================================

CLUSTERING_HEAD = lambda n_clusters: encoder.Encoder(layers=[layers.Dense(n_units=n_clusters)])


ablmvssl_noisymnist_mviic = NoisyMNISTExperiment(
    model_config=mviic.MvIIC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(10),
        overclustering_head_config=None,
        n_overclustering_heads=0,
        loss_config=mviic.MvIICLoss(
            funcs="IICClustering",
            weights=None
        )
    ),
    wandb_tags="ablation,mv-ssl",
)


ablmvssl_caltech7_mviic = Caltech7Experiment(
    model_config=mviic.MvIIC(
        encoder_configs=CALTECH_ENCODERS,
        clustering_head_config=CLUSTERING_HEAD(7),
        overclustering_head_config=None,
        n_overclustering_heads=0,
        loss_config=mviic.MvIICLoss(
            funcs="IICClustering",
            weights=None
        )
    ),
    wandb_tags="ablation,mv-ssl",
)
