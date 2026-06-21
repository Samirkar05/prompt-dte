import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import DownloadConfig, load_dataset
from PIL import Image
import numpy as np


class HFKMNISTDataset(Dataset):
    """Wraps the HuggingFace tanganke/kmnist dataset to behave like torchvision KMNIST."""

    CLASSES = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # HF stores the image as a PIL Image under the 'image' key
        img = sample['image']

        # Torchvision KMNIST is grayscale — ensure mode is 'L'
        if img.mode != 'L':
            img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = sample['label']
        return img, label


class KMNIST:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),  # kept for interface compatibility
        batch_size=128,
        num_workers=6,
    ):
        # Load from HF cache (no re-download if already cached)
        ds = load_dataset(
            "tanganke/kmnist",
            cache_dir=location,
            download_config=DownloadConfig(local_files_only=True),
        )

        self.train_dataset = HFKMNISTDataset(ds['train'], transform=preprocess)
        self.test_dataset  = HFKMNISTDataset(ds['test'],  transform=preprocess)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = HFKMNISTDataset.CLASSES
