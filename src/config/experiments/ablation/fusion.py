from config.experiments.base_experiments import *
from config.templates.models import eamc, simvc_comvc, custom
from config.templates.fusion import Concat


# ======================================================================================================================
# NoisyMNIST
# ======================================================================================================================

ablfusion_noisymnist_eamc = NoisyMNISTExperiment(
    model_config=eamc.EAMC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        attention_config=None,
        fusion_config=Concat(),
        loss_config=eamc.EAMCLoss(
            weights=[1, 1, 1, 10, 1],
            funcs="DDC1|DDC2Flipped|DDC3|EAMCGenerator|EAMCDiscriminator",
        )
    ),
    wandb_tags="ablation,fusion"
)


ablfusion_noisymnist_simvc = NoisyMNISTExperiment(
    model_config=simvc_comvc.SiMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_noisymnist_comvc = NoisyMNISTExperiment(
    model_config=simvc_comvc.CoMVC(
        encoder_configs=NOISY_MNIST_ENCODERS,
        fusion_config=Concat(),
        loss_config=simvc_comvc.CoMVCLoss(
            contrast_adaptive_weight=False
        )
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_noisymnist_sae = NoisyMNISTExperiment(
    model_config=custom.CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        loss_config=custom.CAELoss(funcs="DDC1|DDC2|DDC3|MSE"),
        pre_train_loss_config=custom.CAELoss(funcs="MSE"),
        projector_config=None,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_noisymnist_cae = NoisyMNISTExperiment(
    model_config=custom.CAE(
        encoder_configs=NOISY_MNIST_ENCODERS,
        decoder_configs=NOISY_MNIST_DECODERS,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)


# ======================================================================================================================
# Caltech7
# ======================================================================================================================

ablfusion_caltech7_eamc = Caltech7Experiment(
    model_config=eamc.EAMC(
        encoder_configs=CALTECH_ENCODERS,
        attention_config=None,
        fusion_config=Concat(),
        loss_config=eamc.EAMCLoss(
            weights=[1, 1, 1, 10, 1],
            funcs="DDC1|DDC2Flipped|DDC3|EAMCGenerator|EAMCDiscriminator",
        )
    ),
    wandb_tags="ablation,fusion"
)


ablfusion_caltech7_simvc = Caltech7Experiment(
    model_config=simvc_comvc.SiMVC(
        encoder_configs=CALTECH_ENCODERS,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_caltech7_comvc = Caltech7Experiment(
    model_config=simvc_comvc.CoMVC(
        encoder_configs=CALTECH_ENCODERS,
        fusion_config=Concat(),
        loss_config=simvc_comvc.CoMVCLoss(
            contrast_adaptive_weight=False
        )
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_caltech7_sae = Caltech7Experiment(
    model_config=custom.CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        loss_config=custom.CAELoss(funcs="DDC1|DDC2|DDC3|MSE"),
        pre_train_loss_config=custom.CAELoss(funcs="MSE"),
        projector_config=None,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)

ablfusion_caltech7_cae = Caltech7Experiment(
    model_config=custom.CAE(
        encoder_configs=CALTECH_ENCODERS,
        decoder_configs=CALTECH_DECODERS,
        fusion_config=Concat(),
    ),
    wandb_tags="ablation,fusion"
)
