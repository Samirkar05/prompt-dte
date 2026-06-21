import torch
from src.datasets.cifar100cont import CIFAR100


def get_dataloader(dataset, is_train, args, image_encoder=None):
    return dataset.train_loader if is_train else dataset.test_loader


def get_dataset(dataset_name, preprocess, location, batch_size=128, num_workers=4,
                val_fraction=0.1, max_val_samples=5000, task_id=0):
    dataset = CIFAR100(
        preprocess, location=location, batch_size=batch_size,
        num_workers=num_workers, task_id=task_id
    )
    return dataset
