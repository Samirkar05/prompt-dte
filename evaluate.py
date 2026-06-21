import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from pipeline_utils import (
    average_vision_checkpoints,
    bootstrap_prompt_learning,
    build_normal_head,
    classification_head_from_payload,
    configured_device,
    create_vision_encoder,
    dataset_location,
    dataset_spec,
    evaluate_classifier,
    evaluation_results_path,
    load_config,
    load_vision_checkpoint,
    merged_vision_path,
    model_spec,
    prompt_head_from_contexts,
    prompt_head_path,
    prompt_learner_path,
    upsert_jsonl,
    vision_checkpoint_path,
    vision_encoder_from_payload,
)


def comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot, VisionFT, DTE, or weight-averaged DTE encoders."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--encoder",
        choices=["ZeroShot", "VisionFT", "DTE", "DTEWeightAverage"],
        required=True,
    )
    parser.add_argument(
        "--head",
        choices=["normal", "prompt", "average-prompts"],
        default="normal",
    )
    parser.add_argument("--prompt-run", default=None)
    parser.add_argument("--prompt-runs", type=comma_list, default=None)
    parser.add_argument("--datasets", type=comma_list, default=None)
    parser.add_argument("--merge-datasets", type=comma_list, default=None)
    parser.add_argument("--prompt-source-datasets", type=comma_list, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-save-merged", action="store_true")
    return parser.parse_args()


def selected_dataset_specs(
    config: Dict[str, Any], names: Optional[Sequence[str]]
) -> List[Dict[str, Any]]:
    if names is None:
        return list(config["datasets"])
    return [dataset_spec(config, name) for name in names]


def prompt_context_paths(
    config: Dict[str, Any],
    model: Dict[str, Any],
    source_datasets: Sequence[Dict[str, Any]],
    run_ids: Sequence[str],
) -> List[Path]:
    paths = []
    for source_dataset in source_datasets:
        for run_id in run_ids:
            path = prompt_learner_path(config, model, source_dataset, run_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing prompt context checkpoint: {path}")
            paths.append(path)
    return paths


def load_encoder_for_dataset(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    encoder_mode: str,
    prompt_run: Optional[str],
    device: torch.device,
) -> torch.nn.Module:
    if encoder_mode == "ZeroShot":
        encoder, clip_model = create_vision_encoder(config, model)
        del clip_model
        return encoder.to(device)
    if encoder_mode == "VisionFT":
        path = vision_checkpoint_path(config, model, dataset, "VisionFT")
    elif encoder_mode == "DTE":
        if prompt_run is None:
            raise ValueError("encoder=DTE requires --prompt-run.")
        path = vision_checkpoint_path(config, model, dataset, "DTE", prompt_run)
    else:
        raise ValueError(f"Unsupported dataset-specific encoder mode: {encoder_mode}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing vision checkpoint: {path}")
    return load_vision_checkpoint(config, model, path, device)


def load_head_for_dataset(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    head_mode: str,
    prompt_run: Optional[str],
    averaged_context_paths: Sequence[Path],
    device: torch.device,
) -> torch.nn.Module:
    data_location = dataset_location(config, dataset)
    if head_mode == "normal":
        return build_normal_head(
            config,
            model,
            dataset["eval_dataset"],
            device,
            data_location=data_location,
        ).to(device)
    if head_mode == "prompt":
        if prompt_run is None:
            raise ValueError("head=prompt requires --prompt-run.")
        path = prompt_head_path(config, model, dataset, prompt_run)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prompt head checkpoint: {path}")
        return classification_head_from_payload(torch.load(path, map_location="cpu")).to(device)
    return prompt_head_from_contexts(
        config,
        model,
        dataset["eval_dataset"],
        averaged_context_paths,
        device,
        data_location=data_location,
    ).to(device)


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    bootstrap_prompt_learning(config)
    model = model_spec(config, args.model)
    device = configured_device(config)
    evaluation_config = config.get("evaluation", {})
    target_datasets = selected_dataset_specs(config, args.datasets)
    batch_size = int(args.batch_size or evaluation_config.get("batch_size", 128))

    prompt_runs = args.prompt_runs
    if prompt_runs is None:
        if args.prompt_run is not None:
            prompt_runs = [args.prompt_run]
        else:
            prompt_runs = evaluation_config.get(
                "prompt_average_runs",
                [run["id"] for run in config["prompt_training"]["runs"]],
            )
    source_prompt_datasets = selected_dataset_specs(
        config,
        args.prompt_source_datasets
        or evaluation_config.get("prompt_average_source_datasets"),
    )
    averaged_context_paths = (
        prompt_context_paths(config, model, source_prompt_datasets, prompt_runs)
        if args.head == "average-prompts"
        else []
    )

    merged_encoder = None
    merge_sources = []
    merged_output = None
    if args.encoder == "DTEWeightAverage":
        if args.prompt_run is None:
            raise ValueError("encoder=DTEWeightAverage requires --prompt-run.")
        source_datasets = selected_dataset_specs(
            config, args.merge_datasets or evaluation_config.get("merge_datasets")
        )
        merge_sources = [
            vision_checkpoint_path(config, model, dataset, "DTE", args.prompt_run)
            for dataset in source_datasets
        ]
        missing = [path for path in merge_sources if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing DTE checkpoints required for averaging:\n"
                + "\n".join(str(path) for path in missing)
            )
        if not args.no_save_merged:
            merged_output = merged_vision_path(config, model, args.prompt_run)
        payload = average_vision_checkpoints(
            config,
            model,
            merge_sources,
            output_path=merged_output,
        )
        merged_encoder = vision_encoder_from_payload(config, model, payload, device)

    for dataset in target_datasets:
        encoder = (
            merged_encoder
            if merged_encoder is not None
            else load_encoder_for_dataset(
                config,
                model,
                dataset,
                args.encoder,
                args.prompt_run,
                device,
            )
        )
        head = load_head_for_dataset(
            config,
            model,
            dataset,
            args.head,
            args.prompt_run,
            averaged_context_paths,
            device,
        )
        metrics = evaluate_classifier(
            config,
            model,
            dataset["eval_dataset"],
            encoder,
            head,
            batch_size,
            data_location=dataset_location(config, dataset),
        )
        record = {
            "model": model["name"],
            "model_id": model["id"],
            "dataset": dataset["name"],
            "eval_dataset": dataset["eval_dataset"],
            "encoder": args.encoder,
            "head": args.head,
            "prompt_run": args.prompt_run,
            "prompt_runs_averaged": prompt_runs if args.head == "average-prompts" else None,
            "prompt_sources": (
                [str(path) for path in averaged_context_paths]
                if args.head == "average-prompts"
                else None
            ),
            "vision_merge_sources": [str(path) for path in merge_sources] or None,
            "merged_checkpoint": str(merged_output) if merged_output else None,
            "batch_size": batch_size,
            "top1": metrics["top1"],
            "samples": metrics["samples"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        upsert_jsonl(
            evaluation_results_path(config, model),
            record,
            key_fields=("model_id", "dataset", "encoder", "head", "prompt_run"),
        )
        print(
            f"{model['id']} {dataset['eval_dataset']} encoder={args.encoder} "
            f"head={args.head} top1={100.0 * metrics['top1']:.2f}%"
        )


if __name__ == "__main__":
    main()
