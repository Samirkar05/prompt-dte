import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100

NUM_TASKS = 10
CLASSES_PER_TASK = 10  # 100 / 10


def get_task_class_indices(task_id):
    """Return the 10 original CIFAR-100 class indices for this task (0-indexed task_id)."""
    start = task_id * CLASSES_PER_TASK
    return list(range(start, start + CLASSES_PER_TASK))


class TaskDataset(torch.utils.data.Dataset):
    """Wraps a CIFAR-100 dataset, keeping only the classes for one task and remapping labels to 0–9."""

    def __init__(self, base_dataset, task_class_indices):
        self.base_dataset = base_dataset
        class_set = set(task_class_indices)
        # Fast index selection via .targets (no iteration over images)
        self.indices = [i for i, t in enumerate(base_dataset.targets) if t in class_set]
        self.label_map = {orig: new for new, orig in enumerate(task_class_indices)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.base_dataset[self.indices[idx]]
        return image, self.label_map[label]


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=16,
                 task_id=0):

        full_train = PyTorchCIFAR100(root=location, download=False, train=True,  transform=preprocess)
        full_test  = PyTorchCIFAR100(root=location, download=False, train=False, transform=preprocess)

        task_class_indices = get_task_class_indices(task_id)
        all_classnames     = full_train.classes  # alphabetically sorted, matches integer labels

        # The 10 class names for this task (in remapped order 0–9)
        self.classnames         = [all_classnames[i] for i in task_class_indices]
        self.task_class_indices = task_class_indices  # original CIFAR-100 indices, useful for logging

        print(f"Task {task_id} classes ({len(self.classnames)}): {self.classnames}")

        self.train_dataset = TaskDataset(full_train, task_class_indices)
        self.test_dataset  = TaskDataset(full_test,  task_class_indices)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
