from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def download_cifar10(data_dir: str = "data/raw"):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    # trochvision crreates cifar-10-bateches-py under root
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)

def make_loaders(data_dir="data/raw", batch_size=128, num_workers=2):
    train_ds = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=get_transforms(train=True)
    )

    test_ds = datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=get_transforms(train=False)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

