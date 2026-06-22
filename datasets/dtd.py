import os
from pathlib import Path

import torch
import torchvision.datasets as datasets
from PIL import Image


class DTDFileList(torch.utils.data.Dataset):
    def __init__(self, images_root, split_file, class_to_idx, transform=None):
        self.images_root = Path(images_root)
        self.transform = transform
        self.class_to_idx = dict(class_to_idx)
        self.classes = [
            classname
            for classname, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        ]
        with Path(split_file).open("r", encoding="utf-8") as input_file:
            relative_paths = [line.strip() for line in input_file if line.strip()]
        self.samples = []
        for relative_path in relative_paths:
            relative_path = relative_path.removeprefix("images/")
            classname = Path(relative_path).parts[0]
            if classname not in self.class_to_idx:
                raise ValueError(f"Unknown DTD class in {split_file}: {classname}")
            image_path = self.images_root / relative_path
            if not image_path.is_file():
                raise FileNotFoundError(f"DTD split image not found: {image_path}")
            self.samples.append((image_path, self.class_to_idx[classname]))
        self.targets = [target for _, target in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        # Data loading code
        traindir = os.path.join(location, 'dtd', 'train')
        valdir = os.path.join(location, 'dtd', 'test')    #CHRISTOS COMMENT - CHANGED DEFAULT FROM 'val' to 'test' BECAUSE WITH VAL THE ACCURACY WAS 98% when testing vis_ftuned

        self.train_dataset = datasets.ImageFolder(
            traindir, transform=preprocess)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]


class DTDVal(DTD):
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16):
        dataset_root = Path(location) / "dtd"
        images_root = dataset_root / "images"
        labels_root = dataset_root / "labels"
        classnames = sorted(path.name for path in images_root.iterdir() if path.is_dir())
        class_to_idx = {classname: index for index, classname in enumerate(classnames)}
        self.train_dataset = DTDFileList(
            images_root,
            labels_root / "train1.txt",
            class_to_idx,
            transform=preprocess,
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.test_dataset = DTDFileList(
            images_root,
            labels_root / "val1.txt",
            class_to_idx,
            transform=preprocess,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.classnames = [classname.replace('_', ' ') for classname in classnames]
