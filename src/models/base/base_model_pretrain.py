from lib.loss import Loss
from models.base.base_model import BaseModel


class BaseModelPreTrain(BaseModel):
    def init_losses(self):
        # Main loss
        self.main_loss = Loss(self.cfg.loss_config, self)
        self.get_loss = self.get_main_loss
        self.optimizer_config = self.cfg.optimizer_config

        # Optional pre-training loss
        if self.cfg.pre_train_loss_config is not None:
            self.pre_train_loss = Loss(self.cfg.pre_train_loss_config, self)
        else:
            self.pre_train_loss = None

    @property
    def requires_pre_train(self):
        return self.pre_train_loss is not None

    def get_main_loss(self):
        return self.main_loss(self)

    def get_pre_train_loss(self):
        return self.pre_train_loss(self)

    def init_pre_train(self):
        assert self.requires_pre_train, "Method `init_pre_train` should not be called when model does not require " \
                                        "pre-training."
        self.get_loss = self.get_pre_train_loss
        self.optimizer_config = self.cfg.pre_train_optimizer_config

    def init_fine_tune(self):
        self.get_loss = self.get_main_loss
        self.optimizer_config = self.cfg.optimizer_config
