import os
import wandb
import warnings
import pytorch_lightning as pl
from copy import deepcopy

import config
from lib.loggers import fix_cmat
from lib.wandb_utils import get_default_run_info


def evaluate(net, ckpt_path, loader, logger):
    # Define a separate trainer here, so we don't get unwanted val_* stuff in the results.
    eval_trainer = pl.Trainer(logger=logger, progress_bar_refresh_rate=0, gpus=config.GPUS)
    if not ckpt_path:
        warnings.warn(f"No 'best' checkpoint found at: '{ckpt_path}'.")
        ckpt_path = None

    results = eval_trainer.test(model=net, test_dataloaders=loader, ckpt_path=ckpt_path, verbose=False)
    assert len(results) == 1
    return results[0]


def log_best_run(val_logs_list, test_logs_list, cfg, experiment_name, tag):
    run_info = get_default_run_info(experiment_name, tag, "best", cfg)
    os.makedirs(run_info.dir, exist_ok=True)
    
    wandb_run = wandb.init(
        project=run_info.project,
        group=run_info.group,
        name=run_info.name,
        dir=run_info.dir,
        id=run_info.id,
        tags=run_info.tags,
        config=run_info.cfg,
        reinit=True
    )

    best_run = None
    best_loss = float("inf")
    for run, logs in enumerate(val_logs_list):
        tot_loss = logs[f"val_loss/{cfg.best_loss_term}"]
        if tot_loss < best_loss:
            best_run = run
            best_loss = tot_loss

    def _log_best(set_type, best_logs):
        best_logs = deepcopy(best_logs)
        best_logs["is_best"] = True
        best_logs["best_run"] = best_run

        if f"{set_type}_metrics/cmat" not in best_logs:
            fix_cmat(best_logs, set_type=set_type)

        for key, value in best_logs.items():
            wandb_run.summary[f"summary/{key}"] = value

    best_val_logs = val_logs_list[best_run]
    best_test_logs = test_logs_list[best_run]

    _log_best("val", best_val_logs)
    _log_best("test", best_test_logs)

    wandb_run.finish()
    return best_val_logs, best_test_logs
