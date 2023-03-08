from config.templates.experiment import Experiment
from config.templates.models.simvc_comvc import SiMVC, CoMVC, CoMVCLoss
from config.templates.models.custom import CAEKM, CAEKMLoss, CAE, CAELoss
from config.templates.dataset import Dataset
from config.templates import encoder


ENCODERS = [encoder.Encoder(input_size=(1, 7, 7), layers="cnn_tiny") for _ in range(12)]
DECODERS = [encoder.Encoder(input_size=None, layers="cnn_tiny_decoder") for _ in range(12)]


class PatchedMnistExperiment(Experiment):
    n_views: int = 12
    n_clusters: int = 3


def _model_config(model_name, views):
    encoders = [ENCODERS[v] for v in views]
    decoders = [DECODERS[v] for v in views]

    if model_name == "simvc":
        cfg = SiMVC(encoder_configs=encoders)
    elif model_name == "comvc":
        cfg = CoMVC(encoder_configs=encoders)
    elif model_name == "comvcnoad":
        cfg = CoMVC(
            encoder_configs=encoders,
            loss_config=CoMVCLoss(
                contrast_adaptive_weight=False
            )
        )

    elif model_name == "saekm":
        cfg = CAEKM(
            encoder_configs=encoders,
            decoder_configs=decoders,
            loss_config=CAEKMLoss(funcs="MSE"),
            projector_config=None,
        )
    elif model_name == "caekm":
        cfg = CAEKM(
            encoder_configs=encoders,
            decoder_configs=decoders,
        )

    elif model_name == "sae":
        cfg = CAE(
            encoder_configs=encoders,
            decoder_configs=decoders,
            loss_config=CAELoss(
                funcs="DDC1|DDC2|DDC3|MSE",
                weights=[1, 1, 1, 0.1],
            ),
            pre_train_loss_config=CAELoss(funcs="MSE", weights=None),
        )

    elif model_name == "cae":
        cfg = CAE(
            encoder_configs=encoders,
            decoder_configs=decoders,
        )

    else:
        raise RuntimeError()

    return cfg


def _experiment(model_name, order, views):
    return PatchedMnistExperiment(
        dataset_config=Dataset(name="patchedmnist", select_views=views),
        model_config=_model_config(model_name, views),
        n_views=len(views),
        wandb_tags=f"incviews,{order}"
    )


# ======================================================================================================================
# simvc
# ======================================================================================================================

incviews2wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1])
incviews3wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_simvc = _experiment("simvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7])
incviews3bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_simvc = _experiment("simvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# comvc
# ======================================================================================================================

incviews2wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1])
incviews3wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_comvc = _experiment("comvc", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7])
incviews3bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_comvc = _experiment("comvc", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# comvcnoad
# ======================================================================================================================

incviews2wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1])
incviews3wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_comvcnoad = _experiment("comvcnoad", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7])
incviews3bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_comvcnoad = _experiment("comvcnoad", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# saekm
# ======================================================================================================================

incviews2wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1])
incviews3wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_saekm = _experiment("saekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7])
incviews3bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_saekm = _experiment("saekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# caekm
# ======================================================================================================================

incviews2wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1])
incviews3wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_caekm = _experiment("caekm", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7])
incviews3bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_caekm = _experiment("caekm", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# sae
# ======================================================================================================================

incviews2wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1])
incviews3wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_sae = _experiment("sae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7])
incviews3bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_sae = _experiment("sae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])


# ======================================================================================================================
# cae
# ======================================================================================================================

incviews2wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1])
incviews3wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2])
incviews4wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5])
incviews5wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6])
incviews6wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9])
incviews7wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10])
incviews8wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11])
incviews9wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3])
incviews10wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4])
incviews11wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7])
incviews12wb_patchedmnist_cae = _experiment("cae", "worst-best", [0, 1, 2, 5, 6, 9, 10, 11, 3, 4, 7, 8])

incviews2bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7])
incviews3bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4])
incviews4bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3])
incviews5bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11])
incviews6bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10])
incviews7bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9])
incviews8bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6])
incviews9bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5])
incviews10bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2])
incviews11bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1])
incviews12bw_patchedmnist_cae = _experiment("cae", "best-worst", [8, 7, 4, 3, 11, 10, 9, 6, 5, 2, 1, 0])