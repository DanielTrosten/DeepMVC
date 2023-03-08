from config.experiments.base_experiments import CALTECH_ENCODERS, CALTECH_DECODERS, Caltech7Experiment
from config.templates.models.simvc_comvc import SiMVC, CoMVC, CoMVCLoss
from config.templates.models.custom import CAEKM, CAEKMLoss, CAE, CAELoss
from config.templates.dataset import Dataset


def _model_config(model_name, views):
    encoders = [CALTECH_ENCODERS[v] for v in views]
    decoders = [CALTECH_DECODERS[v] for v in views]

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
    return Caltech7Experiment(
        dataset_config=Dataset(name="caltech7", select_views=views),
        model_config=_model_config(model_name, views),
        n_views=len(views),
        wandb_tags=f"incviews,{order}"
    )


# ======================================================================================================================
# SiMVC
# ======================================================================================================================

incviews2wb_caltech7_simvc = _experiment("simvc", "worst-best", [2, 0])
incviews3wb_caltech7_simvc = _experiment("simvc", "worst-best", [2, 0, 1])
incviews4wb_caltech7_simvc = _experiment("simvc", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_simvc = _experiment("simvc", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_simvc = _experiment("simvc", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_simvc = _experiment("simvc", "best-worst", [4, 3])
incviews3bw_caltech7_simvc = _experiment("simvc", "best-worst", [4, 3, 5])
incviews4bw_caltech7_simvc = _experiment("simvc", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_simvc = _experiment("simvc", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_simvc = _experiment("simvc", "best-worst", [4, 3, 5, 1, 0, 2])

# ======================================================================================================================
# CoMVC
# ======================================================================================================================

incviews2wb_caltech7_comvc = _experiment("comvc", "worst-best", [2, 0])
incviews3wb_caltech7_comvc = _experiment("comvc", "worst-best", [2, 0, 1])
incviews4wb_caltech7_comvc = _experiment("comvc", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_comvc = _experiment("comvc", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_comvc = _experiment("comvc", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_comvc = _experiment("comvc", "best-worst", [4, 3])
incviews3bw_caltech7_comvc = _experiment("comvc", "best-worst", [4, 3, 5])
incviews4bw_caltech7_comvc = _experiment("comvc", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_comvc = _experiment("comvc", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_comvc = _experiment("comvc", "best-worst", [4, 3, 5, 1, 0, 2])


# ======================================================================================================================
# CoMVC without adaptive weight
# ======================================================================================================================

incviews2wb_caltech7_comvcnoad = _experiment("comvcnoad", "worst-best", [2, 0])
incviews3wb_caltech7_comvcnoad = _experiment("comvcnoad", "worst-best", [2, 0, 1])
incviews4wb_caltech7_comvcnoad = _experiment("comvcnoad", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_comvcnoad = _experiment("comvcnoad", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_comvcnoad = _experiment("comvcnoad", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_comvcnoad = _experiment("comvcnoad", "best-worst", [4, 3])
incviews3bw_caltech7_comvcnoad = _experiment("comvcnoad", "best-worst", [4, 3, 5])
incviews4bw_caltech7_comvcnoad = _experiment("comvcnoad", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_comvcnoad = _experiment("comvcnoad", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_comvcnoad = _experiment("comvcnoad", "best-worst", [4, 3, 5, 1, 0, 2])

# ======================================================================================================================
# SAEKM
# ======================================================================================================================
incviews2wb_caltech7_saekm = _experiment("saekm", "worst-best", [2, 0])
incviews3wb_caltech7_saekm = _experiment("saekm", "worst-best", [2, 0, 1])
incviews4wb_caltech7_saekm = _experiment("saekm", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_saekm = _experiment("saekm", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_saekm = _experiment("saekm", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_saekm = _experiment("saekm", "best-worst", [4, 3])
incviews3bw_caltech7_saekm = _experiment("saekm", "best-worst", [4, 3, 5])
incviews4bw_caltech7_saekm = _experiment("saekm", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_saekm = _experiment("saekm", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_saekm = _experiment("saekm", "best-worst", [4, 3, 5, 1, 0, 2])

# ======================================================================================================================
# CAEKM
# ======================================================================================================================
incviews2wb_caltech7_caekm = _experiment("caekm", "worst-best", [2, 0])
incviews3wb_caltech7_caekm = _experiment("caekm", "worst-best", [2, 0, 1])
incviews4wb_caltech7_caekm = _experiment("caekm", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_caekm = _experiment("caekm", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_caekm = _experiment("caekm", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_caekm = _experiment("caekm", "best-worst", [4, 3])
incviews3bw_caltech7_caekm = _experiment("caekm", "best-worst", [4, 3, 5])
incviews4bw_caltech7_caekm = _experiment("caekm", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_caekm = _experiment("caekm", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_caekm = _experiment("caekm", "best-worst", [4, 3, 5, 1, 0, 2])


# ======================================================================================================================
# SAE
# ======================================================================================================================
incviews2wb_caltech7_sae = _experiment("sae", "worst-best", [2, 0])
incviews3wb_caltech7_sae = _experiment("sae", "worst-best", [2, 0, 1])
incviews4wb_caltech7_sae = _experiment("sae", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_sae = _experiment("sae", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_sae = _experiment("sae", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_sae = _experiment("sae", "best-worst", [4, 3])
incviews3bw_caltech7_sae = _experiment("sae", "best-worst", [4, 3, 5])
incviews4bw_caltech7_sae = _experiment("sae", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_sae = _experiment("sae", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_sae = _experiment("sae", "best-worst", [4, 3, 5, 1, 0, 2])

# ======================================================================================================================
# CAE
# ======================================================================================================================
incviews2wb_caltech7_cae = _experiment("cae", "worst-best", [2, 0])
incviews3wb_caltech7_cae = _experiment("cae", "worst-best", [2, 0, 1])
incviews4wb_caltech7_cae = _experiment("cae", "worst-best", [2, 0, 1, 5])
incviews5wb_caltech7_cae = _experiment("cae", "worst-best", [2, 0, 1, 5, 3])
incviews6wb_caltech7_cae = _experiment("cae", "worst-best", [2, 0, 1, 5, 3, 4])

incviews2bw_caltech7_cae = _experiment("cae", "best-worst", [4, 3])
incviews3bw_caltech7_cae = _experiment("cae", "best-worst", [4, 3, 5])
incviews4bw_caltech7_cae = _experiment("cae", "best-worst", [4, 3, 5, 1])
incviews5bw_caltech7_cae = _experiment("cae", "best-worst", [4, 3, 5, 1, 0])
incviews6bw_caltech7_cae = _experiment("cae", "best-worst", [4, 3, 5, 1, 0, 2])
