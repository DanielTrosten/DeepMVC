import torch as th
import torchvision
import torchvision.transforms as transforms

import config

RAW_DIR = config.DATA_DIR / "raw"


def _torchvision_dataset(dataset_class, means, stds, splits=None, custom_transforms=tuple()):
    img_transforms = list(custom_transforms) + [transforms.ToTensor(), transforms.Normalize(means, stds)]
    transform = transforms.Compose(img_transforms)

    if splits is None:
        datasets = [
            dataset_class(root=config.DATA_DIR / "raw", train=True, download=True, transform=transform),
            dataset_class(root=config.DATA_DIR / "raw", train=False, download=True, transform=transform),
        ]
    else:
        datasets = [dataset_class(root=config.DATA_DIR / "raw", split=split, download=True, transform=transform)
                    for split in splits]

    dataset = th.utils.data.ConcatDataset(datasets)
    loader = th.utils.data.DataLoader(dataset, batch_size=len(dataset))
    data, labels = list(loader)[0]
    return data.numpy(), labels.numpy()


def mnist(custom_transforms=tuple()):
    return _torchvision_dataset(torchvision.datasets.MNIST, means=(0.5,), stds=(0.5,),
                                        custom_transforms=custom_transforms)


def fashion_mnist(custom_transforms=tuple()):
    return _torchvision_dataset(torchvision.datasets.FashionMNIST, means=(0.5,), stds=(0.5,),
                                custom_transforms=custom_transforms)


