import torch as th
import torch.nn as nn
from torchvision.transforms import transforms


class Augmenter(nn.Module):
    def __init__(self, cfg, input_size):
        super(Augmenter, self).__init__()
        self.cfg = cfg
        self.input_size = input_size

    def augment_and_collate(self, samples):
        raise NotImplementedError()

    def forward(self, batch):
        raise NotImplementedError()


class IdentityAugmenter(Augmenter):
    def __init__(self):
        super(IdentityAugmenter, self).__init__(None, None)

    def augment_and_collate(self, samples):
        return th.stack(samples, dim=0)

    def forward(self, batch):
        return batch


class BaseImagerAugmenter(Augmenter):
    def __init__(self, cfg, input_size):
        super(BaseImagerAugmenter, self).__init__(cfg, input_size)
        self.transforms = None

    @staticmethod
    def _compile_transforms(transforms):
        # return th.jit.script(transforms)
        return transforms

    def augment_and_collate(self, samples):
        augmented = [self.transforms(img) for img in samples]
        stacked = th.stack(augmented, dim=0)
        # assert not np.isnan(stacked.numpy()).any()
        return stacked

    def forward(self, batch):
        augmented = [self.transforms(img) for img in batch]
        stacked = th.stack(augmented, dim=0)
        return stacked


class GrayImageAugmenter(BaseImagerAugmenter):
    def __init__(self, cfg, input_size):
        super(GrayImageAugmenter, self).__init__(cfg, input_size)
        _transforms = nn.Sequential(
            transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(kernel_size=3)]), p=0.5),
            transforms.RandomResizedCrop(size=input_size[1:], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
        )
        self.transforms = self._compile_transforms(_transforms)


class VectorAugmenter(Augmenter):
    def __init__(self, cfg, input_size):
        super(VectorAugmenter, self).__init__(cfg, input_size)
        self.augmenters = [
            self._identity,
            self._gauss_noise,
            self._dropout_noise,
        ]
        self.n_augmenters = len(self.augmenters)

    @staticmethod
    def _identity(inp):
        return inp

    def _gauss_noise(self, inp):
        noise = th.normal(mean=0.0, std=self.cfg.gauss_std, size=inp.size()).type_as(inp)
        return inp + noise

    def _dropout_noise(self, inp):
        prob = (1 - self.cfg.dropout_prob) * th.ones_like(inp).type_as(inp)
        return inp * th.bernoulli(prob)

    def augment_and_collate(self, samples):
        stacked = th.stack(samples, dim=0)
        all_augmented = th.stack([aug(stacked) for aug in self.augmenters], dim=0)
        batch_size = len(samples)
        augmentation_idx = th.randint(self.n_augmenters, size=(batch_size,))
        augmented = all_augmented[augmentation_idx, th.arange(batch_size)]
        return augmented

    def forward(self, batch):
        all_augmented = th.stack([aug(batch) for aug in self.augmenters], dim=0)
        batch_size = batch.size(0)
        augmentation_idx = th.randint(self.n_augmenters, size=(batch_size,))
        augmented = all_augmented[augmentation_idx, th.arange(batch_size)]
        return augmented


class Augmenters(nn.Module):
    AUGMENTER_CLASSES = {
        "GrayImageAugmenter": GrayImageAugmenter,
        "VectorAugmenter": VectorAugmenter,
    }

    def __init__(self, cfgs, input_sizes):
        super(Augmenters, self).__init__()
        self.augmenters = nn.ModuleList([self._get_augmenter(cfg, s) for cfg, s in zip(cfgs, input_sizes)])
        self.n_views = len(cfgs)

    def _get_augmenter(self, cfg, input_size):
        if cfg is None:
            return IdentityAugmenter()
        return self.AUGMENTER_CLASSES[cfg.class_name](cfg, input_size)

    def augment_and_collate(self, samples):
        # Transpose nested 'samples' list
        samples = [list(x) for x in zip(*samples)]

        # Collate original views as first elements of output batch
        out = [th.stack(samples[i], dim=0) for i in range(self.n_views)]

        # Apply augmenters to views
        for i in range(self.n_views):
            collated = self.augmenters[i].augment_and_collate(samples[i])
            out.append(collated)

        # Stack the rest of the samples as-is.
        for j in range(self.n_views, len(samples)):
            collated = th.stack(samples[j], dim=0)
            out.append(collated)

        return tuple(out)

    def forward(self, batches):
        return [aug(batch) for aug, batch in zip(self.augmenters, batches)]
