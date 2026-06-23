import argparse
import copy
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from pipeline_utils import (
    bootstrap_prompt_learning,
    build_normal_head,
    classification_head_from_payload,
    configured_device,
    create_vision_encoder,
    dataset_location,
    dataset_spec,
    ImageClassifier,
    load_config,
    model_spec,
    prompt_head_path,
    runtime_args,
    save_vision_checkpoint,
    set_seed,
    stable_seed_offset,
    upsert_jsonl,
    vision_checkpoint_path,
    vision_metrics_path,
    vision_results_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DTE or ordinary vision encoder.")
    parser.add_argument("--config", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--DTE", dest="mode", action="store_const", const="DTE")
    mode.add_argument("--VisionFT", dest="mode", action="store_const", const="VisionFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-run", default=None)
    parser.add_argument("--output-mode", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _evaluate(
    classifier: torch.nn.Module,
    dataset_object: Any,
    args: Any,
    device: torch.device,
) -> Dict[str, float]:
    from src.datasets.common import get_dataloader, maybe_dictionarize

    classifier.eval()
    loader = get_dataloader(dataset_object, is_train=False, args=args, image_encoder=None)
    loss_function = torch.nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss_sum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = classifier(images)
            loss_sum += loss_function(logits, labels).item()
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
            batches += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "loss": loss_sum / batches if batches else 0.0,
    }


def train(
    config: Dict[str, Any],
    mode: str,
    model_id: str,
    dataset_name: str,
    prompt_run: Optional[str],
    force: bool,
    output_mode: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> Path:
    bootstrap_prompt_learning(config)
    from src.datasets.common import get_dataloader, maybe_dictionarize
    from src.datasets.registry import get_dataset
    from src.utils import LabelSmoothing, cosine_lr
    from flop_tracking import TrainingFlopTracker, format_flops

    model = model_spec(config, model_id)
    dataset = dataset_spec(config, dataset_name)
    if mode == "DTE" and not prompt_run:
        raise ValueError("--DTE requires --prompt-run.")
    checkpoint_mode = output_mode or mode
    output_path = vision_checkpoint_path(config, model, dataset, checkpoint_mode, prompt_run)
    if output_path.is_file() and not force:
        print(f"Skipping existing {mode} checkpoint: {output_path}")
        return output_path

    set_seed(
        config,
        offset=stable_seed_offset(mode, model_id, dataset_name, prompt_run),
    )
    device = configured_device(config)
    training_config = config.get("vision_training", {})
    batch_size = int(training_config.get("batch_size", 128))
    data_location = dataset_location(config, dataset)
    args = runtime_args(config, model, batch_size, dataset=dataset)
    vision_encoder, clip_model = create_vision_encoder(config, model)
    vision_encoder = vision_encoder.to(device)

    source_prompt_head = None
    if mode == "DTE":
        source_prompt_head = prompt_head_path(config, model, dataset, str(prompt_run))
        if not source_prompt_head.is_file():
            raise FileNotFoundError(f"Missing prompt head for DTE training: {source_prompt_head}")
        classification_head = classification_head_from_payload(
            torch.load(source_prompt_head, map_location="cpu")
        )
    else:
        classification_head = build_normal_head(
            config,
            model,
            dataset["train_dataset"],
            device,
            clip_model=clip_model,
            data_location=data_location,
        )
    del clip_model

    classification_head = classification_head.to(device)
    for parameter in classification_head.parameters():
        parameter.requires_grad_(False)
    for parameter in vision_encoder.parameters():
        parameter.requires_grad_(True)
    classifier = ImageClassifier(vision_encoder, classification_head).to(device)

    dataset_object = get_dataset(
        dataset["train_dataset"],
        vision_encoder.train_preprocess,
        location=data_location,
        batch_size=batch_size,
        num_workers=int(config.get("runtime", {}).get("num_workers", 16)),
    )
    loader = get_dataloader(dataset_object, is_train=True, args=args, image_encoder=None)
    epochs = int(dataset["vision_epochs"])
    learning_rate = float(training_config.get("lr", 1e-5))
    warmup_length = int(training_config.get("warmup_length", 500))
    parameters = [parameter for parameter in vision_encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(training_config.get("weight_decay", 0.1)),
    )
    scheduler = cosine_lr(optimizer, learning_rate, warmup_length, epochs * len(loader))
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    loss_function = (
        LabelSmoothing(label_smoothing)
        if label_smoothing > 0
        else torch.nn.CrossEntropyLoss()
    )
    tracker = TrainingFlopTracker(enabled=bool(training_config.get("track_flops", True)))

    best_accuracy = -1.0
    best_epoch = -1
    iterations_to_best = 0
    total_iterations = 0
    training_seconds = 0.0
    seconds_to_best = 0.0
    best_flops = tracker.snapshot()
    started_at = datetime.now(timezone.utc).isoformat()
    print_every = int(config.get("runtime", {}).get("print_every", 100))

    stop_training = False
    for epoch in range(epochs):
        classifier.train()
        classifier.classification_head.eval()
        train_loss = 0.0
        train_batches = 0
        for batch_index, batch in enumerate(loader):
            if max_iterations is not None and total_iterations >= max_iterations:
                stop_training = True
                break
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
            train_loss += loss.item()
            train_batches += 1
            training_seconds += max(0.0, time.time() - iteration_started - profile_seconds)
            total_iterations += 1
            tracker.record_iteration(int(labels.size(0)))
            if step % print_every == 0:
                batch_flops = tracker.profile_for_batch(int(labels.size(0)))
                print(
                    f"{checkpoint_mode} {dataset['name']} epoch={epoch + 1}/{epochs} "
                    f"batch={batch_index}/{len(loader)} loss={loss.item():.6f} "
                    f"FLOPs/iter={format_flops(batch_flops['flops_per_iteration'])}",
                    flush=True,
                )

        if train_batches == 0:
            break
        validation = _evaluate(classifier, dataset_object, args, device)
        mean_train_loss = train_loss / train_batches if train_batches else 0.0
        print(
            f"{checkpoint_mode} validation {dataset['name']} epoch={epoch + 1}/{epochs} "
            f"accuracy={100.0 * validation['accuracy']:.2f}% "
            f"train_loss={mean_train_loss:.6f} val_loss={validation['loss']:.6f}",
            flush=True,
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_epoch = epoch
            iterations_to_best = total_iterations
            seconds_to_best = training_seconds
            best_flops = copy.deepcopy(tracker.snapshot())
            save_vision_checkpoint(
                output_path,
                vision_encoder,
                model,
                {
                    "mode": mode,
                    "checkpoint_mode": checkpoint_mode,
                    "dataset": dataset["name"],
                    "train_dataset": dataset["train_dataset"],
                    "prompt_run": prompt_run,
                    "source_prompt_head": str(source_prompt_head) if source_prompt_head else None,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                },
            )
        if stop_training:
            break

    finished_at = datetime.now(timezone.utc).isoformat()
    if best_epoch < 0:
        raise RuntimeError(f"{checkpoint_mode} training completed without a best checkpoint.")
    flop_metrics = tracker.snapshot()
    metrics = {
        "model": model["name"],
        "model_id": model["id"],
        "dataset": dataset["name"],
        "train_dataset": dataset["train_dataset"],
        "training_method": checkpoint_mode,
        "base_training_method": mode,
        "prompt_run": prompt_run,
        "source_prompt_head": str(source_prompt_head) if source_prompt_head else None,
        "checkpoint_path": str(output_path),
        "lr": learning_rate,
        "batch_size": batch_size,
        "warmup_length": warmup_length,
        "best_epoch": best_epoch,
        "epochs_to_best": best_epoch + 1,
        "total_epochs": epochs,
        "best_accuracy": best_accuracy,
        "iterations_to_best": iterations_to_best,
        "total_iterations": total_iterations,
        "max_iterations": max_iterations,
        "time_to_best_seconds": seconds_to_best,
        "total_training_seconds": training_seconds,
        "flops_to_best": best_flops.get("total_training_flops"),
        "forward_flops_to_best": best_flops.get("total_forward_flops"),
        "backward_flops_to_best": best_flops.get("total_backward_flops"),
        "run_started_at": started_at,
        "run_finished_at": finished_at,
    }
    metrics.update(flop_metrics)
    write_json(vision_metrics_path(output_path), metrics)
    upsert_jsonl(
        vision_results_path(config, model),
        metrics,
        key_fields=("model_id", "dataset", "training_method", "prompt_run"),
    )
    print(f"Saved best {checkpoint_mode} encoder to {output_path}")
    return output_path


def main() -> None:
    cli_args = parse_args()
    config, _ = load_config(cli_args.config)
    train(
        config,
        cli_args.mode,
        cli_args.model,
        cli_args.dataset,
        cli_args.prompt_run,
        cli_args.force,
        output_mode=cli_args.output_mode,
        max_iterations=cli_args.max_iterations,
    )


if __name__ == "__main__":
    main()
