import os
from pathlib import Path

import torch
import torchvision.datasets as datasets


class KMNIST:
    CLASSNAMES = ['お', 'き', 'す', 'つ', 'な', 'は', 'ま', 'や', 'れ', 'を']

    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):
        roots = [Path(location), Path(location) / "KMNIST", Path(location) / "kmnist"]
        errors = []
        for root in roots:
            try:
                train_dataset = datasets.KMNIST(
                    root=str(root), download=False, train=True, transform=preprocess
                )
                test_dataset = datasets.KMNIST(
                    root=str(root), download=False, train=False, transform=preprocess
                )
                break
            except RuntimeError as exc:
                errors.append(f"{root}: {exc}")
        else:
            raise FileNotFoundError(
                "Local Torchvision KMNIST files were not found. Checked:\n"
                + "\n".join(errors)
            )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.classnames = self.CLASSNAMES
