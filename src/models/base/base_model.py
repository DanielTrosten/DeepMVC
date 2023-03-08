import numpy as np
import torch as th
import pytorch_lightning as pl

import helpers
from lib.metrics import calc_metrics
from lib.encoder import EncoderList
from lib.loss import Loss


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, flatten_encoder_output=True):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.n_views = cfg.n_views
        self.data_module = None

        self.encoders = EncoderList(cfg.encoder_configs, flatten_output=flatten_encoder_output)

        self.loss = None
        self.init_losses()

    def init_losses(self):
        self.loss = Loss(self.cfg.loss_config, self)

    @property
    def requires_pre_train(self):
        return False

    def attach_data_module(self, data_module):
        self.data_module = data_module

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_loss(self):
        return self.loss(self)

    @staticmethod
    def _optimizer_from_cfg(cfg, params):
        if cfg.opt_type == "adam":
            optimizer = th.optim.Adam(params, lr=cfg.learning_rate)
        elif cfg.opt_type == "sgd":
            optimizer = th.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.sgd_momentum)
        else:
            raise RuntimeError()

        if getattr(cfg, "scheduler_config", None) is None:
            # We didn't get a scheduler-config
            return optimizer

        s_cfg = cfg.scheduler_config

        if s_cfg.warmup_epochs is not None:
            if s_cfg.warmup_epochs > 0:
                # Linear warmup scheduler
                warmup_lrs = np.linspace(0, 1, s_cfg.warmup_epochs + 1, endpoint=False)[1:]
                scheduler_lambda = lambda epoch: warmup_lrs[epoch] if epoch < s_cfg.warmup_epochs else 1.0
            else:
                scheduler_lambda = lambda epoch: 1.0
            scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)
        else:
            # Multiplicative decay scheduler
            assert (s_cfg.step_size is not None) and (s_cfg.gamma is not None)
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=s_cfg.step_size, gamma=s_cfg.gamma)

        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._optimizer_from_cfg(self.cfg.optimizer_config, self.parameters())

    def _log_dict(self, dct, prefix, sep="/", ignore_keys=tuple()):
        if prefix:
            prefix += sep
        for key, value in dct.items():
            if key not in ignore_keys:
                self.log(prefix + key, float(value))

    def split_batch(self, batch, **_):
        assert len(batch) == (self.n_views + 1), f"Invalid number of tensors in batch ({len(batch)}) for model " \
                                                   f"{self.__class__.__name__}"
        views = batch[:self.n_views]
        labels = batch[-1]
        return views, labels

    def _train_step(self, batch):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)
        losses = self.get_loss()

        self._log_dict(losses, prefix="train_loss")
        return losses["tot"]

    def training_step(self, batch, idx):
        return self._train_step(batch)

    def _val_test_step(self, batch, idx, prefix):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        pred = self(*inputs)

        # Only evaluate losses on full batches
        if labels.size(0) == self.cfg.batch_size:
            losses = self.get_loss()
            self._log_dict(losses, prefix=f"{prefix}_loss")

        return np.stack((helpers.npy(labels), helpers.npy(pred).argmax(axis=1)), axis=0)

    def _val_test_epoch_end(self, step_outputs, prefix):
        if not isinstance(step_outputs, list):
            step_outputs = [step_outputs]

        labels_pred = np.concatenate(step_outputs, axis=1)
        mtc = calc_metrics(labels=labels_pred[0], pred=labels_pred[1])
        self._log_dict(mtc, prefix=f"{prefix}_metrics")

    def validation_step(self, batch, idx):
        return self._val_test_step(batch, idx, "val")

    def validation_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, "val")

    def test_step(self, batch, idx):
        return self._val_test_step(batch, idx, self.test_prefix)

    def test_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, self.test_prefix)
