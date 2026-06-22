# Prompt DTE Pipeline

This project reuses the dataset, dataloader, head, optimizer, and FLOP utilities from a configurable `prompt_learning` checkout. It does not modify `prompt_learning` or `model-merging`.

## Modes

- Prompt training learns shared context vectors and materializes a classification head.
- `--DTE` trains one visual encoder with one selected prompt head and stores that encoder independently.
- `--VisionFT` trains a visual encoder with the normal zero-shot classification head.
- Evaluation can average prompt context vectors before creating a target-dataset head.
- Evaluation can equal-weight average DTE visual encoder state dictionaries across datasets.

## Configuration

Copy `config.example.json` and set every path under `paths`. Relative paths are resolved from the configuration file.

For the `cgeorgakilas` environment, `config.cgeorgakilas.json` is prepared with:

- Dataset root: `/data/125-1/users/cgeorgakilas/task_datasets`
- OpenCLIP cache: `/home-local/cgeorgakilas/.cache/open_clip`
- Local dataset definitions: `./datasets`

Confirm that the OpenCLIP cache directory name matches the target machine; the supplied screenshot was truncated after `.cache/openc...`.

Datasets are never hardcoded in the scripts. Each configured dataset supplies:

- `name`: pipeline identifier.
- `train_dataset`: registered dataset used for training and validation.
- `eval_dataset`: registered dataset used for final evaluation.
- `prompt_epochs`: prompt-learning budget.
- `vision_epochs`: DTE and VisionFT budget.

New datasets work after their implementations, templates, and registry entries are available under the configured `dataset_code_root`.

The copied dataset registry is mounted ahead of `prompt_learning/src/datasets`, so these local definitions are used without modifying the original project. Dataset downloads are disabled. FER2013 additionally requires the Hugging Face `datasets` package and an existing offline cache under its configured data root. KMNIST uses local Torchvision files.

The four copied loaders requiring new official-validation handling are DTD, Flowers102, PCAM, and RenderedSST2. Their `*Val` names use official validation data rather than splitting training data. Existing explicit validation support for EuroSAT and RESISC45 is also retained.

Each dataset may define an optional `data_root` to override `paths.data_location`. This is necessary if the datasets do not share one common parent directory.

Validate the target-user setup before launching GPU work:

```bash
python scheduler.py --config config.cgeorgakilas.json --validate-data
```

This verifies the local registry override, dataset roots, train/validation/test splits, OpenCLIP cache visibility, optional dependencies, and writable output directories without downloading datasets.

The copied 20-dataset registry is loaded from `paths.dataset_code_root` ahead of the original registry. See `DATASETS.md` for expected directory names, official validation aliases, offline behavior, and the preflight command.

For the `cgeorgakilas` setup, use `config.cgeorgakilas.json`. It targets:

- Dataset root: `/data/125-1/users/cgeorgakilas/task_datasets`
- Source checkout: `/home-local/cgeorgakilas/task_vectors-main`
- OpenCLIP cache: `/home-local/cgeorgakilas/.cache/open_clip`
- Checkpoints: `/data/125-1/users/cgeorgakilas/samir_stuff/checkpoints`
- Metrics and logs: relative to the cloned project under `./artifacts/`

The project directory itself must be copied to, or made readable and writable from, that account. Install the additions in `requirements-extra.txt` in the same Python environment used by the source checkout.

## Full Workflow

Validate paths, dependencies, dataset layouts, and split aliases before using GPUs:

```bash
python scheduler.py --config config.cgeorgakilas.json --validate-data
```

Run prompt training, DTE training, and VisionFT training in sequence:

```bash
python scheduler.py --config config.json --stage all
```

Run individual stages:

```bash
python scheduler.py --config config.json --stage prompt
python scheduler.py --config config.json --stage DTE
python scheduler.py --config config.json --stage VisionFT
```

Preview all subprocess commands without training:

```bash
python scheduler.py --config config.json --stage all --dry-run
```

Existing artifacts are skipped. Add `--force` to retrain them.
Use `--model`, `--dataset`, and `--prompt-run` to schedule a configured subset.

## Direct Vision Training

Train one DTE encoder from one prompt head:

```bash
python train_vision_encoder.py \
  --config config.json \
  --DTE \
  --model ViT-B-32 \
  --dataset Cars \
  --prompt-run bs128_wu500_lr1e-3
```

Train the normal vision-finetuning baseline:

```bash
python train_vision_encoder.py \
  --config config.json \
  --VisionFT \
  --model ViT-B-32 \
  --dataset Cars
```

## Evaluation

Evaluate dataset-specific DTE encoders with their prompt heads:

```bash
python evaluate.py \
  --config config.json \
  --model ViT-B-32 \
  --encoder DTE \
  --head prompt \
  --prompt-run bs128_wu500_lr1e-3
```

Evaluate ordinary vision finetuning with normal heads:

```bash
python evaluate.py \
  --config config.json \
  --model ViT-B-32 \
  --encoder VisionFT \
  --head normal
```

Equal-weight average all configured DTE visual encoders and average all configured prompt contexts:

```bash
python evaluate.py \
  --config config.json \
  --model ViT-B-32 \
  --encoder DTEWeightAverage \
  --head average-prompts \
  --prompt-run bs128_wu500_lr1e-3
```

Use `--merge-datasets` to control which DTE encoders are merged and `--prompt-source-datasets` or `--prompt-runs` to control prompt averaging.

## Artifacts

All locations derive from the configuration:

- Prompt contexts and heads: `<checkpoints_root>/<model>/prompts/`
- DTE and VisionFT encoders: `<checkpoints_root>/<model>/vision/`
- Merged DTE encoders: `<checkpoints_root>/<model>/merged/`
- Metrics and evaluations: `<results_root>/<model>/`
- Worker logs: `<logs_root>/<model>/`
