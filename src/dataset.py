from pathlib import Path

from torch.utils.data import Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split
from .preprocessing import get_train_transforms,get_test_transforms


def create_datasets(data_dir: str, validation_size: float = 0.2):
    data_path = Path(data_dir)

    train_dir = data_path / "train"
    test_dir = data_path / "test"

    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )

    validation_dataset_full = datasets.ImageFolder(
        root=train_dir,
        transform=get_test_transforms()
    )

    labels = full_train_dataset.targets

    indices = list(range(len(full_train_dataset)))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        stratify=labels,
        random_state=42
    )

    train_dataset = Subset(
        full_train_dataset,
        train_indices
    )

    validation_dataset = Subset(
        validation_dataset_full,
        validation_indices
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_test_transforms()
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset
    )