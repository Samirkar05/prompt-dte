# Dataset Setup

The target configuration uses this common root:

```text
/data/125-1/users/cgeorgakilas/task_datasets
```

The copied loaders resolve the following locations relative to that root:

| Dataset | Expected location or cache usage |
|---|---|
| Cars | `cars/stanford_cars/` |
| CIFAR10 | Torchvision CIFAR10 files under the root |
| CIFAR100 | Torchvision CIFAR100 files under the root |
| DTD | `dtd/images/`, `dtd/labels/train1.txt`, `val1.txt`, and `dtd/test/` |
| EMNIST | `EMNIST/` with Torchvision EMNIST files |
| EuroSAT | `EuroSAT_splits/train/`, `val/`, and `test/` |
| FER2013 | Hugging Face cache rooted at the configured dataset root |
| FashionMNIST | `FashionMNIST/` with Torchvision FashionMNIST files |
| Flowers102 | `flowers102/` with Torchvision train, val, and test split files |
| Food101 | `food101/` with Torchvision Food101 files |
| GTSRB | `gtsrb/` |
| KMNIST | Local Torchvision KMNIST files under the root, `KMNIST/`, or `kmnist/` |
| MNIST | Torchvision MNIST files under the root |
| OxfordIIITPet | `OxfordIIITPet/` |
| PCAM | `PCAM/` with train, val, and test HDF5 files |
| RESISC45 | `resisc/resisc45/` split files and `resisc/NWPU-RESISC45/` images |
| RenderedSST2 | `RenderedSST2/` |
| STL10 | `stl10/` |
| SUN397 | `SUN397/train/` and `SUN397/test/` |
| SVHN | `svhn/` |

If a dataset lives elsewhere, add a dataset-level `data_root` field. It overrides `paths.data_location` only for that dataset.

## Validation Splits

These aliases use official validation data and never derive validation examples from training data:

- `DTDVal`
- `EuroSATVal`
- `Flowers102Val`
- `PCAMVal`
- `RESISC45Val`
- `RenderedSST2Val`

Every other configured `*Val` name deterministically takes 10% of the training set, capped at 5,000 examples, using seed 0.

## Offline Behavior

All Torchvision loaders use `download=False`. FER2013 uses Hugging Face with local-files-only and offline environment variables. The target environment therefore needs the `datasets` Python package even though no network download is performed.

Validate the complete setup as the target user before training:

```bash
python scheduler.py --config config.cgeorgakilas.json --validate-data
```
