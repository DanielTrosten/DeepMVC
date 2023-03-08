import os
import wandb
import helpers
import numpy as np
from collections import namedtuple

import config


WANDB_CONST_VARS = {
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_API_KEY",
}

WANDB_PROJECT = "mvc-framework"
WANDB_SWEEP_ID = os.environ.get("WANDB_SWEEP_ID", None)


def get_experiment_tag():
    tag = os.environ.get("EXPERIMENT_ID", None)
    if tag is None:
        tag = wandb.util.generate_id()
        print(f"Could not find EXPERIMENT_ID in environment variables. Using generated tag '{tag}'.")
    return tag


def get_default_run_info(experiment_name, tag, run, cfg):
    RunInfo = namedtuple("RunInfo", ["project", "id", "name", "group", "dir", "tags", "cfg"])

    cfg = config.dict_from_cfg(cfg)
    cfg["sweep_id"] = WANDB_SWEEP_ID

    if cfg.get("wandb_tags", None) is not None:
        tags = cfg["wandb_tags"].split(",")
    else:
        tags = None

    if isinstance(run, int):
        run = f"run-{run}"

    group = f"{experiment_name}-{tag}"
    name = f"{group}-{run}"
    _dir = helpers.get_save_dir(experiment_name, tag, run)
    os.makedirs(_dir, exist_ok=True)

    return RunInfo(
        project=WANDB_PROJECT,
        group=group,
        name=name,
        id=name,
        dir=str(_dir),
        tags=tags,
        cfg=cfg,
    )


def clear_wandb_env():
    cleared_vars = {}
    for var_name, value in os.environ.items():
        if var_name.startswith("WANDB_") and (var_name not in WANDB_CONST_VARS):
            print(f"Removing environment variable {var_name} = {value}")
            del os.environ[var_name]
            cleared_vars[var_name] = value
    return cleared_vars


def restore_wandb_env(restore_vars):
    for var_name, value in restore_vars.items():
        assert var_name not in WANDB_CONST_VARS, f"Attempted to restore constant variable {var_name}"
        print(f"Restoring environment variable {var_name} = {value}")
        os.environ[var_name] = value


def init_sweep_run(experiment_name, tag, cfg, sweep_vars):
    assert WANDB_SWEEP_ID is not None, "Attempting to initialize a sweep-run, but no WANDB_SWEEP_ID was found in the " \
                                       "environment variables"
    # Restore the wandb-vars to the original state
    restore_wandb_env(sweep_vars)
    run_info = get_default_run_info(experiment_name, tag, "sweep", cfg)
    # Initialize run
    sweep_run = wandb.init(
        id=os.environ["WANDB_RUN_ID"],
        group=run_info.group,
        name=run_info.name,
        config=run_info.cfg,
        dir=run_info.dir,
        tags=run_info.tags,
    )
    return sweep_run


def finalize_sweep_run(sweep_run, all_logs):
    val_logs, test_logs, best_val_logs, best_test_logs = all_logs

    for logs, best in [(val_logs, best_val_logs), (test_logs, best_test_logs)]:
        for key in best.keys():
            if "cmat" in key:
                # Don't log confusion matrix stuff.
                continue
            values = np.array([d[key] for d in logs])
            sweep_run.summary[f"sweep/{key}/best"] = best[key]
            sweep_run.summary[f"sweep/{key}/mean"] = values.mean()
            sweep_run.summary[f"sweep/{key}/std"] = values.std()
            sweep_run.summary[f"sweep/{key}/min"] = values.min()
            sweep_run.summary[f"sweep/{key}/max"] = values.max()

    sweep_run.finish()
