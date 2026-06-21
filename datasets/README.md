# Expected Offline Layout

`location` is `data_root` from the dataset entry when present, otherwise `paths.data_location`.

The copied loaders expect these paths relative to `location`:

| Dataset | Expected path or cache convention |
|---|---|
| Cars | `cars/stanford_cars/` |
| CIFAR10 | `cifar-10-batches-py/` |
| CIFAR100 | `cifar-100-python/` |
| DTD | `dtd/train/`, `dtd/val/`, `dtd/test/` |
| EMNIST | Loader root is `EMNIST/` |
| EuroSAT | `EuroSAT_splits/train/`, `EuroSAT_splits/val/`, `EuroSAT_splits/test/` |
| FER2013 | Existing Hugging Face cache rooted at `location` |
| FashionMNIST | Loader root is `FashionMNIST/` |
| Flowers102 | `flowers102/flowers-102/` |
| Food101 | `food101/food-101/` |
| GTSRB | `gtsrb/` |
| KMNIST | Existing Hugging Face cache rooted at `location` |
| MNIST | `MNIST/` |
| OxfordIIITPet | `OxfordIIITPet/oxford-iiit-pet/` |
| PCAM | `PCAM/pcam/` |
| RESISC45 | `resisc/resisc45/` split files and `resisc/NWPU-RESISC45/` images |
| RenderedSST2 | `RenderedSST2/rendered-sst2/` |
| STL10 | `stl10/stl10_binary/` |
| SUN397 | `SUN397/train/` and `SUN397/test/` |
| SVHN | `svhn/` |

Run the following as the target user to detect layout or permission mismatches:

```bash
python scheduler.py --config config.cgeorgakilas.json --validate-data
```

Automatic downloads are disabled in every copied loader.
