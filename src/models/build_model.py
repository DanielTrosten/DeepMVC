import torch as th

import config
import helpers
from register import MODEL_CLASSES
from data.data_module import DataModule


def fuzzy_model_match(model_name):
    for class_name, model_class in MODEL_CLASSES.items():
        if class_name in model_name:
            return model_class
    raise ValueError(f"Invalid model type: {model_name}")


def strict_model_match(model_name):
    try:
        return MODEL_CLASSES[model_name]
    except KeyError as err:
        raise ValueError(f"Invalid model type: {model_name}") from err


def build_model(model_cfg, skip_load_weights=False, run=None):
    model = strict_model_match(model_cfg.class_name)(model_cfg).to(config.DEVICE, non_blocking=True)

    if (model_cfg.initial_weights is not None) and (not skip_load_weights):
        load_weights(model, run, model_cfg.initial_weights)

    return model


def match_state_dicts(current, new):
    for key, value in current.items():
        if key not in new:
            print(f"Could not find expected key {key} in initial weights file. These weights will be randomly "
                  f"initialized.")
        elif value.size() != new[key].size():
            print(f"Shape mismatch for key {key}: {value.size()} != {new[key].size()}. These weights will be randomly "
                  f"initialized.")
            del new[key]


def load_weights(model, run, initial_weights):
    assert run is not None, "Cannot have run=None when specifying initial weights."
    weights_file = config.INITIAL_WEIGHTS_DIR / initial_weights / f"run-{run}.pt"
    loaded_state_dict = th.load(weights_file)
    
    match_state_dicts(model.state_dict(), loaded_state_dict)

    missing, unexpected = model.load_state_dict(loaded_state_dict, strict=False)

    print(f"Successfully loaded initial weights from {weights_file}")
    if missing:
        print(f"Weights {missing} were not present in the initial weights file.")
    if unexpected:
        print(f"Unexpected weights {unexpected} were present in the initial weights file. These will be ignored.")


def from_file(experiment_name=None, tag=None, run=None, ckpt="best", return_data=False, return_config=False, **kwargs):
    try:
        cfg = config.get_config_from_file(name=experiment_name, tag=tag)
    except FileNotFoundError:
        print("WARNING: Could not get pickled config.")
        cfg = config.get_config_by_name(experiment_name)

    model_dir = helpers.get_save_dir(experiment_name, identifier=tag, run=run)
    if ckpt == "best":
        model_file = "best.ckpt"
    elif isinstance(ckpt, int):
        model_file = f"checkpoint_epoch={str(ckpt).zfill(4)}.ckpt"
    else:
        model_file = ckpt

    model_path = model_dir / model_file
    net = strict_model_match(cfg.model_config.class_name).load_from_checkpoint(str(model_path), cfg=cfg.model_config)
    net.eval()

    out = [net]

    if return_data:
        dm = DataModule(cfg.dataset_config)
        out.append(dm)

    if return_config:
        out.append(cfg)

    if len(out) == 1:
        out = out[0]

    return out
