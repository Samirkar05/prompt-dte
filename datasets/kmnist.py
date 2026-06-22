import os

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class HFKMNISTDataset(Dataset):
    CLASSNAMES = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        sample = self.split[index]
        image = sample['image'].convert('L')
        if self.transform is not None:
            image = self.transform(image)
        return image, sample['label']


class KMNIST:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):
        dataset = load_dataset("tanganke/kmnist", cache_dir=location)
        self.train_dataset = HFKMNISTDataset(dataset['train'], transform=preprocess)
        self.test_dataset = HFKMNISTDataset(dataset['test'], transform=preprocess)
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
        self.classnames = HFKMNISTDataset.CLASSNAMES
