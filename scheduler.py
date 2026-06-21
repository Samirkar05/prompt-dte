import argparse
import copy
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pipeline_utils import (
    bootstrap_prompt_learning,
    configured_device,
    create_clip_model,
    dataset_location,
    dataset_spec,
    load_config,
    model_spec,
    PromptLearner,
    prompt_head_path,
    prompt_learner_path,
    prompt_metrics_path,
    prompt_results_path,
    prompt_run_spec,
    prompt_text_features,
    runtime_args,
    safe_token,
    set_seed,
    stable_seed_offset,
    upsert_jsonl,
    vision_checkpoint_path,
    write_json,
)


@dataclass
class Task:
    label: str
    command: List[str]
    log_path: Path


class PromptClassifier(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module, prompt_learner: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = prompt_text_features(self.clip_model, self.prompt_learner)
        return self.clip_model.logit_scale.exp() * image_features @ text_features.T


def _save_prompt_artifacts(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    run: Dict[str, Any],
    classifier: PromptClassifier,
    metrics: Dict[str, Any],
) -> None:
    learner_path = prompt_learner_path(config, model, dataset, run["id"])
    head_path = prompt_head_path(config, model, dataset, run["id"])
    learner_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ctx": classifier.prompt_learner.ctx.detach().cpu().clone(),
            "n_ctx": classifier.prompt_learner.n_ctx,
            "model": model["name"],
            "dataset": dataset["train_dataset"],
            "run_id": run["id"],
        },
        learner_path,
    )
    classifier.eval()
    classifier.clip_model.eval()
    with torch.no_grad():
        weights = prompt_text_features(classifier.clip_model, classifier.prompt_learner).cpu()
    torch.save(
        {
            "weight": weights,
            "bias": None,
            "normalize": True,
            "logit_scale": float(
                classifier.clip_model.logit_scale.exp().detach().cpu().item()
            ),
            "source_prompt_checkpoint": str(learner_path),
            "model": model["name"],
            "dataset": dataset["train_dataset"],
            "run_id": run["id"],
        },
        head_path,
    )
    write_json(prompt_metrics_path(config, model, dataset, run["id"]), metrics)


def train_prompt_worker(
    config: Dict[str, Any],
    model_id: str,
    dataset_name: str,
    run_id: str,
    force: bool,
) -> None:
    bootstrap_prompt_learning(config)
    from src.datasets.common import get_dataloader, maybe_dictionarize
    from src.datasets.registry import get_dataset
    from src.utils import cosine_lr
    from flop_tracking import TrainingFlopTracker, format_flops

    model = model_spec(config, model_id)
    dataset = dataset_spec(config, dataset_name)
    run = prompt_run_spec(config, run_id)
    output_path = prompt_head_path(config, model, dataset, run_id)
    if output_path.is_file() and not force:
        print(f"Skipping existing prompt head: {output_path}")
        return

    set_seed(config, offset=stable_seed_offset(model_id, dataset_name, run_id))
    device = configured_device(config)
    clip_model, train_preprocess, _ = create_clip_model(config, model)
    clip_model = clip_model.to(device).eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad_(False)

    data_location = dataset_location(config, dataset)
    args = runtime_args(config, model, int(run["batch_size"]), dataset=dataset)
    dataset_object = get_dataset(
        dataset["train_dataset"],
        train_preprocess,
        location=data_location,
        batch_size=int(run["batch_size"]),
        num_workers=int(config.get("runtime", {}).get("num_workers", 16)),
    )
    prompt_learner = PromptLearner(
        dataset_object.classnames,
        clip_model,
        n_ctx=int(config.get("prompt_training", {}).get("n_ctx", 16)),
        device=device,
    ).to(device)
    classifier = PromptClassifier(clip_model, prompt_learner).to(device)
    parameters = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(run["lr"]),
        weight_decay=float(run.get("weight_decay", config["prompt_training"].get("weight_decay", 0.1))),
    )
    loader = get_dataloader(dataset_object, is_train=True, args=args, image_encoder=None)
    epochs = int(dataset["prompt_epochs"])
    scheduler = cosine_lr(
        optimizer,
        float(run["lr"]),
        int(run["warmup_length"]),
        epochs * len(loader),
    )
    loss_function = torch.nn.CrossEntropyLoss()
    tracker = TrainingFlopTracker(
        enabled=bool(config.get("prompt_training", {}).get("track_flops", True))
    )

    best_accuracy = -1.0
    best_epoch = -1
    iterations_to_best = 0
    total_iterations = 0
    training_seconds = 0.0
    seconds_to_best = 0.0
    best_flops = tracker.snapshot()
    best_context = None
    started_at = datetime.now(timezone.utc).isoformat()

    for epoch in range(epochs):
        classifier.train()
        classifier.clip_model.eval()
        for batch_index, batch in enumerate(loader):
            iteration_started = time.time()
            step = batch_index + epoch * len(loader)
            scheduler(step)
            optimizer.zero_grad()
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            profile_seconds = tracker.profile_batch(
                int(labels.size(0)),
                classifier,
                optimizer,
                lambda: loss_function(classifier(images), labels),
            )
            loss = loss_function(classifier(images), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            training_seconds += max(0.0, time.time() - iteration_started - profile_seconds)
            total_iterations += 1
            tracker.record_iteration(int(labels.size(0)))
            if step % int(config.get("runtime", {}).get("print_every", 100)) == 0:
                batch_flops = tracker.profile_for_batch(int(labels.size(0)))
                print(
                    f"Prompt {dataset['name']} epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{len(loader)} loss={loss.item():.6f} "
                    f"FLOPs/iter={format_flops(batch_flops['flops_per_iteration'])}",
                    flush=True,
                )

        classifier.eval()
        classifier.clip_model.eval()
        validation_loader = get_dataloader(
            dataset_object, is_train=False, args=args, image_encoder=None
        )
        correct = 0
        count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = maybe_dictionarize(batch)
                images = batch["images"].to(device)
                labels = batch["labels"].to(device)
                predictions = classifier(images).argmax(dim=1)
                correct += predictions.eq(labels).sum().item()
                count += labels.size(0)
        accuracy = correct / count if count else 0.0
        print(
            f"Prompt validation {dataset['name']} epoch={epoch + 1}/{epochs} "
            f"accuracy={100.0 * accuracy:.2f}%",
            flush=True,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            iterations_to_best = total_iterations
            seconds_to_best = training_seconds
            best_flops = copy.deepcopy(tracker.snapshot())
            best_context = classifier.prompt_learner.ctx.detach().cpu().clone()
            partial_metrics = {
                "model": model["name"],
                "model_id": model["id"],
                "dataset": dataset["name"],
                "train_dataset": dataset["train_dataset"],
                "run_id": run["id"],
                "best_epoch": best_epoch,
                "epochs_to_best": best_epoch + 1,
                "total_epochs": epochs,
                "best_accuracy": best_accuracy,
                "iterations_to_best": iterations_to_best,
            }
            _save_prompt_artifacts(config, model, dataset, run, classifier, partial_metrics)

    finished_at = datetime.now(timezone.utc).isoformat()
    if best_context is None:
        raise RuntimeError("Prompt training completed without a best checkpoint.")
    classifier.prompt_learner.ctx.data.copy_(
        best_context.to(device, dtype=classifier.prompt_learner.ctx.dtype)
    )
    flop_metrics = tracker.snapshot()
    metrics = {
        "model": model["name"],
        "model_id": model["id"],
        "dataset": dataset["name"],
        "train_dataset": dataset["train_dataset"],
        "run_id": run["id"],
        "training_method": "prompt_learning",
        "lr": float(run["lr"]),
        "batch_size": int(run["batch_size"]),
        "warmup_length": int(run["warmup_length"]),
        "best_epoch": best_epoch,
        "epochs_to_best": best_epoch + 1,
        "total_epochs": epochs,
        "best_accuracy": best_accuracy,
        "iterations_to_best": iterations_to_best,
        "total_iterations": total_iterations,
        "time_to_best_seconds": seconds_to_best,
        "total_training_seconds": training_seconds,
        "flops_to_best": best_flops.get("total_training_flops"),
        "forward_flops_to_best": best_flops.get("total_forward_flops"),
        "backward_flops_to_best": best_flops.get("total_backward_flops"),
        "prompt_learner_path": str(prompt_learner_path(config, model, dataset, run_id)),
        "prompt_head_path": str(prompt_head_path(config, model, dataset, run_id)),
        "run_started_at": started_at,
        "run_finished_at": finished_at,
    }
    metrics.update(flop_metrics)
    _save_prompt_artifacts(config, model, dataset, run, classifier, metrics)
    upsert_jsonl(
        prompt_results_path(config, model),
        metrics,
        key_fields=("model_id", "dataset", "run_id"),
    )


def configured_gpus(config: Dict[str, Any]) -> List[Optional[str]]:
    scheduler_config = config.get("scheduler", {})
    if scheduler_config.get("gpus") is not None:
        return [str(gpu) for gpu in scheduler_config["gpus"]]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [token.strip() for token in visible.split(",") if token.strip()]
    if str(config.get("runtime", {}).get("device", "cuda")).startswith("cpu"):
        return [None]
    return [str(index) for index in range(torch.cuda.device_count())]


def run_tasks(
    tasks: List[Task], config: Dict[str, Any], dry_run: bool
) -> None:
    if not tasks:
        return
    if dry_run:
        for task in tasks:
            print(f"[DRY RUN] {task.label}: {' '.join(task.command)}")
        return
    gpus = configured_gpus(config)
    if not gpus:
        raise RuntimeError("No GPUs are available for scheduled tasks.")
    max_parallel = min(
        len(gpus), int(config.get("scheduler", {}).get("max_parallel", len(gpus)))
    )
    available = list(gpus[:max_parallel])
    pending = list(tasks)
    active = []

    while pending or active:
        while pending and available:
            task = pending.pop(0)
            gpu = available.pop(0)
            environment = os.environ.copy()
            if gpu is not None:
                environment["CUDA_VISIBLE_DEVICES"] = gpu
            task.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = task.log_path.open("w", encoding="utf-8")
            print(f"[LAUNCH][GPU {gpu}] {task.label} -> {task.log_path}")
            process = subprocess.Popen(
                task.command,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active.append((process, task, gpu, log_file))

        still_active = []
        failure = None
        for process, task, gpu, log_file in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, task, gpu, log_file))
                continue
            log_file.close()
            available.append(gpu)
            print(f"[DONE][GPU {gpu}] rc={return_code} {task.label}")
            if return_code != 0 and failure is None:
                failure = (task, return_code)
        active = still_active
        if failure is not None:
            for process, _, _, log_file in active:
                process.terminate()
                log_file.close()
            raise RuntimeError(
                f"Scheduled task failed with exit code {failure[1]}: {failure[0].label}"
            )
        if active:
            time.sleep(float(config.get("scheduler", {}).get("poll_seconds", 2.0)))


def prompt_tasks(
    config: Dict[str, Any],
    config_path: Path,
    force: bool,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    run_filter: Optional[str] = None,
) -> List[Task]:
    tasks = []
    script = Path(__file__).resolve()
    logs_root = Path(config["paths"]["logs_root"])
    for model in config["models"]:
        if model_filter is not None and model_filter not in {model["id"], model["name"]}:
            continue
        for dataset in config["datasets"]:
            if dataset_filter is not None and dataset_filter not in {
                dataset["name"],
                dataset["train_dataset"],
            }:
                continue
            for run in config["prompt_training"]["runs"]:
                if run_filter is not None and run_filter != run["id"]:
                    continue
                output = prompt_head_path(config, model, dataset, run["id"])
                if output.is_file() and not force:
                    print(f"[SKIP] prompt {model['id']} {dataset['name']} {run['id']}")
                    continue
                command = [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--prompt-worker",
                    "--model",
                    model["id"],
                    "--dataset",
                    dataset["name"],
                    "--prompt-run",
                    run["id"],
                ]
                if force:
                    command.append("--force")
                label = f"prompt:{model['id']}:{dataset['name']}:{run['id']}"
                log_path = logs_root / model["id"] / f"{safe_token(label)}.log"
                tasks.append(Task(label, command, log_path))
    return tasks


def dte_tasks(
    config: Dict[str, Any],
    config_path: Path,
    force: bool,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    run_filter: Optional[str] = None,
) -> List[Task]:
    tasks = []
    script = Path(__file__).resolve().parent / "train_vision_encoder.py"
    logs_root = Path(config["paths"]["logs_root"])
    run_ids = config.get("vision_training", {}).get(
        "dte_prompt_runs", [run["id"] for run in config["prompt_training"]["runs"]]
    )
    for model in config["models"]:
        if model_filter is not None and model_filter not in {model["id"], model["name"]}:
            continue
        for dataset in config["datasets"]:
            if dataset_filter is not None and dataset_filter not in {
                dataset["name"],
                dataset["train_dataset"],
            }:
                continue
            for run_id in run_ids:
                if run_filter is not None and run_filter != run_id:
                    continue
                output = vision_checkpoint_path(config, model, dataset, "DTE", run_id)
                if output.is_file() and not force:
                    print(f"[SKIP] DTE {model['id']} {dataset['name']} {run_id}")
                    continue
                command = [
                    sys.executable,
                    str(script),
                    "--config",
                    str(config_path),
                    "--DTE",
                    "--model",
                    model["id"],
                    "--dataset",
                    dataset["name"],
                    "--prompt-run",
                    run_id,
                ]
                if force:
                    command.append("--force")
                label = f"DTE:{model['id']}:{dataset['name']}:{run_id}"
                log_path = logs_root / model["id"] / f"{safe_token(label)}.log"
                tasks.append(Task(label, command, log_path))
    return tasks


def visionft_tasks(
    config: Dict[str, Any],
    config_path: Path,
    force: bool,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
) -> List[Task]:
    tasks = []
    script = Path(__file__).resolve().parent / "train_vision_encoder.py"
    logs_root = Path(config["paths"]["logs_root"])
    for model in config["models"]:
        if model_filter is not None and model_filter not in {model["id"], model["name"]}:
            continue
        for dataset in config["datasets"]:
            if dataset_filter is not None and dataset_filter not in {
                dataset["name"],
                dataset["train_dataset"],
            }:
                continue
            output = vision_checkpoint_path(config, model, dataset, "VisionFT")
            if output.is_file() and not force:
                print(f"[SKIP] VisionFT {model['id']} {dataset['name']}")
                continue
            command = [
                sys.executable,
                str(script),
                "--config",
                str(config_path),
                "--VisionFT",
                "--model",
                model["id"],
                "--dataset",
                dataset["name"],
            ]
            if force:
                command.append("--force")
            label = f"VisionFT:{model['id']}:{dataset['name']}"
            log_path = logs_root / model["id"] / f"{safe_token(label)}.log"
            tasks.append(Task(label, command, log_path))
    return tasks


def validate_data_setup(config: Dict[str, Any]) -> None:
    bootstrap_prompt_learning(config)
    import src.datasets.registry as dataset_registry

    expected_registry_root = Path(config["paths"]["dataset_code_root"]).resolve()
    actual_registry = Path(dataset_registry.__file__).resolve()
    if actual_registry.parent != expected_registry_root:
        raise RuntimeError(
            f"Dataset override is inactive: loaded {actual_registry}, expected {expected_registry_root}"
        )

    roots = {
        Path(config["paths"]["data_location"]),
        *{
            Path(dataset["data_root"])
            for dataset in config["datasets"]
            if dataset.get("data_root")
        },
    }
    errors = []
    for root in sorted(roots):
        if not root.is_dir():
            errors.append(f"Dataset root is missing or inaccessible: {root}")
        elif not os.access(root, os.R_OK | os.X_OK):
            errors.append(f"Dataset root is not readable/traversable: {root}")

    cache_root = Path(config["paths"]["openclip_cache_dir"])
    if not cache_root.is_dir():
        errors.append(f"OpenCLIP cache is missing or inaccessible: {cache_root}")
    elif not os.access(cache_root, os.R_OK | os.X_OK):
        errors.append(f"OpenCLIP cache is not readable/traversable: {cache_root}")

    for key in ("checkpoints_root", "results_root", "logs_root"):
        output_root = Path(config["paths"][key])
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Cannot create paths.{key} at {output_root}: {exc}")
            continue
        if not os.access(output_root, os.W_OK | os.X_OK):
            errors.append(f"Output path is not writable/traversable: {output_root}")

    for dataset in config["datasets"]:
        location = dataset_location(config, dataset)
        if dataset.get("validation_source") == "official":
            if dataset["train_dataset"] not in dataset_registry.OFFICIAL_VALIDATION_DATASETS:
                errors.append(
                    f"{dataset['name']}: {dataset['train_dataset']} is not an official validation alias"
                )
                continue
        split_names = [dataset["train_dataset"], dataset["eval_dataset"]]
        for split_name in split_names:
            try:
                loaded = dataset_registry.get_dataset(
                    split_name,
                    None,
                    location=location,
                    batch_size=2,
                    num_workers=0,
                )
                train_size = len(loaded.train_dataset)
                evaluation_size = len(loaded.test_dataset)
                split_kind = (
                    "official-validation"
                    if split_name in dataset_registry.OFFICIAL_VALIDATION_DATASETS
                    else "registered/test-or-derived-validation"
                )
                print(
                    f"[OK] {split_name} root={location} train={train_size} "
                    f"eval={evaluation_size} split={split_kind}"
                )
            except Exception as exc:
                errors.append(f"{split_name} at {location}: {type(exc).__name__}: {exc}")

    if errors:
        print("Dataset setup validation failed:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError(f"Dataset setup has {len(errors)} error(s).")
    print("All configured dataset splits and the OpenCLIP cache are accessible.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven prompt, DTE, and VisionFT scheduler.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "prompt", "DTE", "VisionFT", "vision"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="Validate configured dataset roots and split loaders without training.",
    )
    parser.add_argument("--prompt-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-run", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    if args.validate_data:
        validate_data_setup(config)
        return
    if args.prompt_worker:
        if not args.model or not args.dataset or not args.prompt_run:
            raise ValueError("Prompt worker requires --model, --dataset, and --prompt-run.")
        train_prompt_worker(config, args.model, args.dataset, args.prompt_run, args.force)
        return

    if args.stage in {"all", "prompt"}:
        run_tasks(
            prompt_tasks(
                config,
                config_path,
                args.force,
                args.model,
                args.dataset,
                args.prompt_run,
            ),
            config,
            args.dry_run,
        )
    if args.stage in {"all", "vision", "DTE"}:
        run_tasks(
            dte_tasks(
                config,
                config_path,
                args.force,
                args.model,
                args.dataset,
                args.prompt_run,
            ),
            config,
            args.dry_run,
        )
    if args.stage in {"all", "vision", "VisionFT"}:
        run_tasks(
            visionft_tasks(
                config,
                config_path,
                args.force,
                args.model,
                args.dataset,
            ),
            config,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
