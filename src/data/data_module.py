import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.load import load_dataset
from data.augmenter import Augmenters


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.is_pre_train = False
        self.train_dataset = self.val_dataset = self.test_dataset = None

        self.train_dataset = load_dataset(
            name=cfg.name,
            split="train",
            random_seed=cfg.random_seed,
            n_samples=cfg.n_train_samples,
            select_views=cfg.select_views,
            select_labels=cfg.select_labels,
            label_counts=cfg.train_label_counts,
            to_dataset=True,
            include_index=cfg.include_index,
        )

        self.val_dataset = load_dataset(
            name=cfg.name,
            split="val",
            random_seed=cfg.random_seed,
            n_samples=cfg.n_val_samples,
            select_views=cfg.select_views,
            select_labels=cfg.select_labels,
            label_counts=cfg.val_label_counts,
            to_dataset=True,
            include_index=cfg.include_index,
        ) or self.train_dataset

        self.test_dataset = load_dataset(
            name=cfg.name,
            split="test",
            random_seed=cfg.random_seed,
            n_samples=cfg.n_test_samples,
            select_views=cfg.select_views,
            select_labels=cfg.select_labels,
            label_counts=cfg.test_label_counts,
            to_dataset=True,
            include_index=cfg.include_index,
        ) or self.train_dataset

        # Add data augmentation?
        if cfg.augmenter_configs is not None:
            input_sizes = [tuple(t.size()[1:]) for t in self.train_dataset.tensors]
            self.augmenter = Augmenters(cfgs=cfg.augmenter_configs, input_sizes=input_sizes)
        else:
            self.augmenter = None

        # Number of training batches
        self.n_train_samples = len(self.train_dataset)
        self.n_batches = self.n_train_samples // cfg.batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def _get_batch_size(self):
        if self.is_pre_train and ((pt_batch_size := self.cfg.pre_train_batch_size) is not None):
            batch_size = pt_batch_size
        else:
            batch_size = self.cfg.batch_size
        return batch_size

    def _get_shuffle(self, split):
        shuffle = getattr(self.cfg, f"{split}_shuffle")
        if self.is_pre_train:
            pt_shuffle = getattr(self.cfg, f"pre_train_{split}_shuffle")
            if pt_shuffle is not None:
                shuffle = pt_shuffle
        return shuffle

    def _get_loader(self, split, enable_augmentations, shuffle, drop_last, pin_memory):

        batch_size = self._get_batch_size()
        n_workers = getattr(self.cfg, f"n_{split}_workers")
        dataset = getattr(self, f"{split}_dataset")

        if drop_last is None:
            drop_last = self.cfg.drop_last

        if shuffle is None:
            shuffle = self._get_shuffle(split)

        if enable_augmentations and (self.augmenter is not None):
            collate_fn = self.augmenter.augment_and_collate
        else:
            collate_fn = None

        dataloader_kwargs = dict(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=self.cfg.n_train_workers, pin_memory=pin_memory, collate_fn=collate_fn,
        )
        if n_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.cfg.prefetch_factor

        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self, enable_augmentations=True, shuffle=None, drop_last=None, pin_memory=True):
        return self._get_loader(split="train", enable_augmentations=enable_augmentations, shuffle=shuffle,
                                drop_last=drop_last, pin_memory=pin_memory)

    def val_dataloader(self, enable_augmentations=True, shuffle=None, drop_last=False, pin_memory=True):
        return self._get_loader(split="val", enable_augmentations=enable_augmentations, shuffle=shuffle,
                                drop_last=drop_last, pin_memory=pin_memory)

    def test_dataloader(self, enable_augmentations=True, shuffle=None, drop_last=False, pin_memory=True):
        return self._get_loader(split="test", enable_augmentations=enable_augmentations, shuffle=shuffle,
                                drop_last=drop_last, pin_memory=pin_memory)
