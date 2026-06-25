import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from budget_merging import merge_state
from flop_tracking import format_flops
from pipeline_utils import (
    bootstrap_prompt_learning,
    build_normal_head,
    classification_head_from_payload,
    configured_device,
    create_vision_encoder,
    dataset_location,
    dataset_spec,
    evaluate_classifier,
    load_vision_checkpoint,
    load_config,
    model_root,
    model_spec,
    prompt_head_path,
    prompt_metrics_path,
    prompt_run_spec,
    safe_token,
    upsert_jsonl,
    vision_checkpoint_path,
    vision_encoder_from_payload,
    vision_metrics_path,
    write_json,
)
from scheduler import Task, prompt_tasks, run_tasks, visionft_tasks


APPROACH_EVEN = "even_dataset_budget"
APPROACH_DATASET = "per_dataset_budget"
MODE_EVEN = "DTEBudgetEven"
MODE_DATASET = "DTEBudgetDataset"


def comma_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budgeted prompt/DTE/VisionFT scheduler with merge evaluation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "prompt", "VisionFT", "budget", "evaluate"],
        default="all",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-run", default=None)
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Output namespace to resume or reuse. Defaults to a new timestamp for all/prompt.",
    )
    parser.add_argument("--methods", type=comma_list, default=None)
    parser.add_argument("--pre-val", dest="pre_val", action="store_true", default=None)
    parser.add_argument("--no-pre-val", dest="pre_val", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eval-worker",
        choices=["pre_val", "merged"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--approach",
        choices=[APPROACH_EVEN, APPROACH_DATASET],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--merge-method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-subset", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def selected_models(config: Dict[str, Any], model_filter: Optional[str]) -> List[Dict[str, Any]]:
    if model_filter is None:
        return list(config["models"])
    return [model_spec(config, model_filter)]


def selected_datasets(config: Dict[str, Any], dataset_filter: Optional[str]) -> List[Dict[str, Any]]:
    if dataset_filter is None:
        return list(config["datasets"])
    return [dataset_spec(config, dataset_filter)]


def selected_prompt_run(config: Dict[str, Any], run_filter: Optional[str]) -> str:
    if run_filter is not None:
        prompt_run_spec(config, run_filter)
        return run_filter
    budget_config = config.get("budget_scheduler", {})
    configured = budget_config.get("prompt_run")
    if configured:
        prompt_run_spec(config, configured)
        return configured
    return config["prompt_training"]["runs"][0]["id"]


def selected_methods(config: Dict[str, Any], methods: Optional[Sequence[str]]) -> List[str]:
    configured = methods or config.get("budget_scheduler", {}).get(
        "merge_methods", ["task_arithmetic", "weight_average", "tsv_m", "iso_c"]
    )
    allowed = {"task_arithmetic", "weight_average", "tsv_m", "iso_c"}
    unknown = [method for method in configured if method not in allowed]
    if unknown:
        raise ValueError(f"Unknown merge method(s): {', '.join(unknown)}")
    return list(configured)


def available_experiment_ids(config: Dict[str, Any]) -> List[str]:
    experiment_ids = set()
    for key in ("checkpoints_root", "results_root", "logs_root"):
        root = Path(config["paths"][key]) / "budget_runs"
        if root.is_dir():
            experiment_ids.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(experiment_ids)


def resolve_experiment_id(config: Dict[str, Any], args: argparse.Namespace) -> str:
    configured = config.get("budget_scheduler", {}).get("experiment_id")
    explicit = args.experiment_id or configured
    if explicit:
        return safe_token(explicit)

    if args.stage in {"VisionFT", "budget", "evaluate"}:
        existing = available_experiment_ids(config)
        if existing:
            return existing[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return safe_token(f"budget_{timestamp}")


def apply_experiment_namespace(config: Dict[str, Any], experiment_id: str) -> None:
    budget_config = config.setdefault("budget_scheduler", {})
    budget_config["resolved_experiment_id"] = experiment_id
    if budget_config.get("namespace_outputs", True) is False:
        return
    for key in ("checkpoints_root", "results_root", "logs_root"):
        root = Path(config["paths"][key])
        if len(root.parts) >= 2 and root.parts[-2:] == ("budget_runs", experiment_id):
            config["paths"][key] = str(root)
        else:
            config["paths"][key] = str(root / "budget_runs" / experiment_id)


def materialize_experiment_config(config: Dict[str, Any], dry_run: bool) -> Path:
    experiment_id = config.get("budget_scheduler", {}).get("resolved_experiment_id", "budget")
    path = (
        Path("/tmp") / f"prompt_dte_{safe_token(experiment_id)}.json"
        if dry_run
        else Path(config["paths"]["results_root"]) / "_experiment_config.json"
    )
    write_json(path, config)
    return path


def budget_prompt_run_id(base_run_id: str) -> str:
    return f"{base_run_id}__budget_dataset"


def budget_plan_path(config: Dict[str, Any], model: Dict[str, Any], run_id: str) -> Path:
    return Path(config["paths"]["results_root"]) / safe_token(model["id"]) / f"budget_plan_{safe_token(run_id)}.json"


def budget_results_path(config: Dict[str, Any], model: Dict[str, Any]) -> Path:
    return Path(config["paths"]["results_root"]) / safe_token(model["id"]) / "budget_evaluations.jsonl"


def budget_merged_path(
    config: Dict[str, Any],
    model: Dict[str, Any],
    approach: str,
    method: str,
    run_id: str,
    task_subset: Optional[str] = None,
) -> Path:
    subset_prefix = f"{safe_token(task_subset)}_" if task_subset else ""
    return (
        model_root(config, model)
        / "merged"
        / "budget"
        / safe_token(approach)
        / f"{subset_prefix}{safe_token(method)}_{safe_token(run_id)}.pt"
    )


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return read_json(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def find_evaluation_record(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    approach: str,
    merge_method: str,
    base_prompt_run: str,
    task_subset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for row in read_jsonl(budget_results_path(config, model)):
        if (
            row.get("model_id") == model["id"]
            and row.get("dataset") == dataset["name"]
            and row.get("approach") == approach
            and row.get("merge_method") == merge_method
            and row.get("base_prompt_run") == base_prompt_run
            and row.get("task_subset") == task_subset
        ):
            return row
    return None


def upsert_evaluation_record(
    config: Dict[str, Any],
    model: Dict[str, Any],
    record: Dict[str, Any],
) -> None:
    upsert_jsonl(
        budget_results_path(config, model),
        record,
        key_fields=(
            "model_id",
            "dataset",
            "approach",
            "merge_method",
            "base_prompt_run",
            "task_subset",
        ),
    )


def metric_number(metrics: Dict[str, Any], key: str, path: Path) -> float:
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"Missing metric {key}: {path}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Metric {key} is not finite: {path}")
    return value


def best_backward_flops(metrics: Dict[str, Any], path: Path) -> float:
    return metric_number(metrics, "backward_flops_to_best", path)


def backward_flops_per_iteration(metrics: Dict[str, Any], path: Path) -> float:
    value = metrics.get("backward_flops_per_iteration")
    if value is not None:
        value = float(value)
        if value > 0 and math.isfinite(value):
            return value
    total = best_backward_flops(metrics, path)
    iterations = metric_number(metrics, "iterations_to_best", path)
    if iterations <= 0:
        raise ValueError(f"Cannot infer backward FLOPs/iteration from {path}")
    return total / iterations


def collect_budget_rows(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    run_id: str,
) -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        prompt_path = prompt_metrics_path(config, model, dataset, run_id)
        vision_path = vision_metrics_path(vision_checkpoint_path(config, model, dataset, "VisionFT"))
        if not prompt_path.is_file() or not vision_path.is_file():
            missing = [str(path) for path in (prompt_path, vision_path) if not path.is_file()]
            raise FileNotFoundError(
                "Budget computation requires completed prompt and VisionFT metrics:\n"
                + "\n".join(missing)
            )
        prompt_metrics = read_json(prompt_path)
        vision_metrics = read_json(vision_path)
        rows.append(
            {
                "dataset": dataset["name"],
                "train_dataset": dataset["train_dataset"],
                "prompt_backward_to_best": best_backward_flops(prompt_metrics, prompt_path),
                "visionft_backward_to_best": best_backward_flops(vision_metrics, vision_path),
                "prompt_backward_per_iteration": backward_flops_per_iteration(
                    prompt_metrics, prompt_path
                ),
                "dte_backward_per_iteration": backward_flops_per_iteration(
                    vision_metrics, vision_path
                ),
                "prompt_metrics": str(prompt_path),
                "visionft_metrics": str(vision_path),
            }
        )
    return rows


def iterations_for_budget(budget: float, backward_per_iteration: float) -> int:
    if budget <= 0 or backward_per_iteration <= 0:
        return 0
    return int(math.floor(budget / backward_per_iteration))


def prompt_budget_iterations(budget: float, backward_per_iteration: float) -> int:
    if budget <= 0 or backward_per_iteration <= 0:
        return 0
    return max(1, int(math.floor(budget / backward_per_iteration)))


def build_budget_plan(
    rows: Sequence[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    mean_prompt = sum(row["prompt_backward_to_best"] for row in rows) / len(rows)
    mean_vision = sum(row["visionft_backward_to_best"] for row in rows) / len(rows)
    even_budget = mean_vision - mean_prompt
    plan_rows = []
    for row in rows:
        dataset_budget = row["visionft_backward_to_best"] - row["prompt_backward_to_best"]
        even_iterations = iterations_for_budget(even_budget, row["dte_backward_per_iteration"])
        dataset_iterations = iterations_for_budget(
            dataset_budget, row["dte_backward_per_iteration"]
        )
        negative_prompt_iterations = (
            prompt_budget_iterations(
                row["visionft_backward_to_best"], row["prompt_backward_per_iteration"]
            )
            if dataset_budget < 0
            else 0
        )
        plan_rows.append(
            {
                **row,
                "even_budget": even_budget,
                "even_dte_iterations": even_iterations,
                "dataset_budget": dataset_budget,
                "dataset_dte_iterations": dataset_iterations,
                "dataset_prompt_iterations": negative_prompt_iterations,
                "dataset_prompt_run": (
                    budget_prompt_run_id(run_id) if dataset_budget < 0 else run_id
                ),
                "include_in_dataset_merge": dataset_budget >= 0 and dataset_iterations > 0,
                "negative_budget": dataset_budget < 0,
            }
        )
    return {
        "prompt_run": run_id,
        "mean_prompt_backward_to_best": mean_prompt,
        "mean_visionft_backward_to_best": mean_vision,
        "even_budget": even_budget,
        "datasets": plan_rows,
    }


def budget_task_log(config: Dict[str, Any], model: Dict[str, Any], label: str) -> Path:
    return (
        Path(config["paths"]["logs_root"])
        / safe_token(model["id"])
        / "budget"
        / f"{safe_token(label)}.log"
    )


def budget_dte_tasks(
    config: Dict[str, Any],
    config_path: Path,
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    approach: str,
    force: bool,
) -> List[Task]:
    mode = MODE_EVEN if approach == APPROACH_EVEN else MODE_DATASET
    iteration_key = (
        "even_dte_iterations" if approach == APPROACH_EVEN else "dataset_dte_iterations"
    )
    script = Path(__file__).resolve().parent / "train_vision_encoder.py"
    tasks = []
    plan_by_dataset = {row["dataset"]: row for row in plan["datasets"]}
    for dataset in datasets:
        row = plan_by_dataset[dataset["name"]]
        iterations = int(row[iteration_key])
        if iterations <= 0:
            continue
        output = vision_checkpoint_path(config, model, dataset, mode, plan["prompt_run"])
        if output.is_file() and not force:
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
            dataset["train_dataset"],
            "--prompt-run",
            plan["prompt_run"],
            "--output-mode",
            mode,
            "--max-iterations",
            str(iterations),
        ]
        if force:
            command.append("--force")
        label = f"{mode}:{model['id']}:{dataset['name']}:{plan['prompt_run']}:{iterations}"
        tasks.append(Task(label, command, budget_task_log(config, model, label)))
    return tasks


def budget_prompt_tasks(
    config: Dict[str, Any],
    config_path: Path,
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    force: bool,
) -> List[Task]:
    script = Path(__file__).resolve().parent / "scheduler.py"
    tasks = []
    plan_by_dataset = {row["dataset"]: row for row in plan["datasets"]}
    output_run = budget_prompt_run_id(plan["prompt_run"])
    for dataset in datasets:
        row = plan_by_dataset[dataset["name"]]
        iterations = int(row["dataset_prompt_iterations"])
        if not row["negative_budget"] or iterations <= 0:
            continue
        output = prompt_head_path(config, model, dataset, output_run)
        if output.is_file() and not force:
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
            dataset["train_dataset"],
            "--prompt-run",
            plan["prompt_run"],
            "--output-prompt-run",
            output_run,
            "--max-iterations",
            str(iterations),
        ]
        if force:
            command.append("--force")
        label = f"PromptBudget:{model['id']}:{dataset['name']}:{output_run}:{iterations}"
        tasks.append(Task(label, command, budget_task_log(config, model, label)))
    return tasks


def refresh_progress_for_scope(config: Dict[str, Any], args: argparse.Namespace, run_id: str) -> None:
    if args.dry_run:
        return
    for model in selected_models(config, args.model):
        refresh_progress_table(config, model, selected_datasets(config, args.dataset), run_id)


def run_prompt_stage(
    config: Dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    run_id: str,
) -> None:
    print("[1/4] Prompt learning")
    run_tasks(
        prompt_tasks(config, config_path, args.force, args.model, args.dataset, run_id),
        config,
        args.dry_run,
    )
    refresh_progress_for_scope(config, args, run_id)


def run_visionft_stage(
    config: Dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    print("[2/4] VisionFT")
    run_tasks(
        visionft_tasks(config, config_path, args.force, args.model, args.dataset),
        config,
        args.dry_run,
    )
    refresh_progress_for_scope(config, args, run_id=selected_prompt_run(config, args.prompt_run))


def run_budget_stage(
    config: Dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    run_id: str,
) -> None:
    print("[3/4] Budgeted DTE")
    for model in selected_models(config, args.model):
        datasets = selected_datasets(config, args.dataset)
        plan = build_budget_plan(collect_budget_rows(config, model, datasets, run_id), run_id)
        if not args.dry_run:
            write_json(budget_plan_path(config, model, run_id), plan)
        even_tasks = budget_dte_tasks(
            config, config_path, model, datasets, plan, APPROACH_EVEN, args.force
        )
        dataset_tasks = budget_dte_tasks(
            config, config_path, model, datasets, plan, APPROACH_DATASET, args.force
        )
        prompt_tasks_for_negative = budget_prompt_tasks(
            config, config_path, model, datasets, plan, args.force
        )
        print(
            f"{model['id']} budget: even={plan['even_budget']:.3e} "
            f"DTE(even)={len(even_tasks)} DTE(dataset)={len(dataset_tasks)} "
            f"prompt-rebudget={len(prompt_tasks_for_negative)}"
        )
        if not args.dry_run:
            refresh_progress_table(config, model, datasets, run_id)
        run_tasks(even_tasks + dataset_tasks + prompt_tasks_for_negative, config, args.dry_run)
        if not args.dry_run:
            refresh_progress_table(config, model, datasets, run_id)


def source_paths_for_approach(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    approach: str,
) -> List[Path]:
    if approach == APPROACH_EVEN:
        mode = MODE_EVEN
        source_datasets = list(datasets)
    else:
        mode = MODE_DATASET
        included = {
            row["dataset"]
            for row in plan["datasets"]
            if bool(row["include_in_dataset_merge"])
        }
        source_datasets = [dataset for dataset in datasets if dataset["name"] in included]
    return [
        vision_checkpoint_path(config, model, dataset, mode, plan["prompt_run"])
        for dataset in source_datasets
    ]


def prompt_run_for_evaluation(plan: Dict[str, Any], dataset: Dict[str, Any], approach: str) -> str:
    if approach == APPROACH_EVEN:
        return str(plan["prompt_run"])
    for row in plan["datasets"]:
        if row["dataset"] == dataset["name"]:
            return str(row["dataset_prompt_run"])
    return str(plan["prompt_run"])


def evaluation_record_payload(
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    approach: str,
    merge_method: str,
    base_prompt_run: str,
    prompt_run: Optional[str],
    batch_size: int,
    metrics: Dict[str, Any],
    task_subset: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    record = {
        "model": model["name"],
        "model_id": model["id"],
        "dataset": dataset["name"],
        "eval_dataset": dataset["eval_dataset"],
        "approach": approach,
        "merge_method": merge_method,
        "prompt_run": prompt_run,
        "base_prompt_run": base_prompt_run,
        "task_subset": task_subset,
        "batch_size": batch_size,
        "top1": metrics["top1"],
        "samples": metrics["samples"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    record.update(extra)
    return record


def evaluate_visionft_baseline(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    base_prompt_run: str,
    batch_size: int,
    device: torch.device,
    force: bool,
) -> bool:
    if not force and find_evaluation_record(
        config, model, dataset, "baseline", "vision_ft", base_prompt_run
    ):
        return False
    checkpoint_path = vision_checkpoint_path(config, model, dataset, "VisionFT")
    if not checkpoint_path.is_file():
        print(f"[SKIP] VisionFT pre-val {dataset['name']}: missing {checkpoint_path}")
        return False
    encoder = load_vision_checkpoint(config, model, checkpoint_path, device)
    head = build_normal_head(
        config,
        model,
        dataset["eval_dataset"],
        device,
        data_location=dataset_location(config, dataset),
    ).to(device)
    metrics = evaluate_classifier(
        config,
        model,
        dataset["eval_dataset"],
        encoder,
        head,
        batch_size,
        data_location=dataset_location(config, dataset),
    )
    upsert_evaluation_record(
        config,
        model,
        evaluation_record_payload(
            model,
            dataset,
            "baseline",
            "vision_ft",
            base_prompt_run,
            None,
            batch_size,
            metrics,
            checkpoint_path=str(checkpoint_path),
        ),
    )
    return True


def evaluate_prompt_baseline(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    approach: str,
    plan: Dict[str, Any],
    batch_size: int,
    device: torch.device,
    force: bool,
) -> bool:
    prompt_run = prompt_run_for_evaluation(plan, dataset, approach)
    if not force and find_evaluation_record(
        config, model, dataset, approach, "prompt_learning", plan["prompt_run"]
    ):
        return False
    head_path = prompt_head_path(config, model, dataset, prompt_run)
    if not head_path.is_file():
        print(f"[SKIP] prompt pre-val {approach} {dataset['name']}: missing {head_path}")
        return False
    if not force:
        for row in read_jsonl(budget_results_path(config, model)):
            if (
                row.get("model_id") == model["id"]
                and row.get("dataset") == dataset["name"]
                and row.get("merge_method") == "prompt_learning"
                and row.get("base_prompt_run") == plan["prompt_run"]
                and row.get("prompt_run") == prompt_run
            ):
                copied = dict(row)
                copied["approach"] = approach
                copied["copied_from_approach"] = row.get("approach")
                upsert_evaluation_record(config, model, copied)
                return True
    encoder, clip_model = create_vision_encoder(config, model)
    del clip_model
    encoder = encoder.to(device)
    head = classification_head_from_payload(torch.load(head_path, map_location="cpu")).to(device)
    metrics = evaluate_classifier(
        config,
        model,
        dataset["eval_dataset"],
        encoder,
        head,
        batch_size,
        data_location=dataset_location(config, dataset),
    )
    upsert_evaluation_record(
        config,
        model,
        evaluation_record_payload(
            model,
            dataset,
            approach,
            "prompt_learning",
            plan["prompt_run"],
            prompt_run,
            batch_size,
            metrics,
            prompt_head_path=str(head_path),
        ),
    )
    return True


def evaluate_dte_baseline(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    approach: str,
    plan: Dict[str, Any],
    batch_size: int,
    device: torch.device,
    force: bool,
) -> bool:
    mode = MODE_EVEN if approach == APPROACH_EVEN else MODE_DATASET
    prompt_run = prompt_run_for_evaluation(plan, dataset, approach)
    if not force and find_evaluation_record(
        config, model, dataset, approach, "dte", plan["prompt_run"]
    ):
        return False
    checkpoint_path = vision_checkpoint_path(config, model, dataset, mode, plan["prompt_run"])
    head_path = prompt_head_path(config, model, dataset, prompt_run)
    if not checkpoint_path.is_file():
        print(f"[SKIP] DTE pre-val {approach} {dataset['name']}: missing {checkpoint_path}")
        return False
    if not head_path.is_file():
        print(f"[SKIP] DTE pre-val {approach} {dataset['name']}: missing {head_path}")
        return False
    encoder = load_vision_checkpoint(config, model, checkpoint_path, device)
    head = classification_head_from_payload(torch.load(head_path, map_location="cpu")).to(device)
    metrics = evaluate_classifier(
        config,
        model,
        dataset["eval_dataset"],
        encoder,
        head,
        batch_size,
        data_location=dataset_location(config, dataset),
    )
    upsert_evaluation_record(
        config,
        model,
        evaluation_record_payload(
            model,
            dataset,
            approach,
            "dte",
            plan["prompt_run"],
            prompt_run,
            batch_size,
            metrics,
            checkpoint_path=str(checkpoint_path),
            prompt_head_path=str(head_path),
        ),
    )
    return True


def run_pre_validation(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    force: bool,
) -> None:
    device = configured_device(config)
    batch_size = int(config.get("evaluation", {}).get("batch_size", 128))
    counters = {"vision_ft": 0, "prompt_learning": 0, "dte": 0}
    for dataset in datasets:
        if evaluate_visionft_baseline(
            config, model, dataset, plan["prompt_run"], batch_size, device, force
        ):
            counters["vision_ft"] += 1
            refresh_results_table(config, model, datasets, plan["prompt_run"])
        for approach in (APPROACH_EVEN, APPROACH_DATASET):
            if evaluate_prompt_baseline(
                config, model, dataset, approach, plan, batch_size, device, force
            ):
                counters["prompt_learning"] += 1
                refresh_results_table(config, model, datasets, plan["prompt_run"])
            if evaluate_dte_baseline(
                config, model, dataset, approach, plan, batch_size, device, force
            ):
                counters["dte"] += 1
                refresh_results_table(config, model, datasets, plan["prompt_run"])
    print(
        f"{model['id']} pre-val evaluated: "
        f"VisionFT={counters['vision_ft']} Prompt={counters['prompt_learning']} DTE={counters['dte']}"
    )


def evaluate_merged_method(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    approach: str,
    method: str,
    force: bool,
    task_subset_filter: Optional[str] = None,
) -> None:
    device = configured_device(config)
    batch_size = int(config.get("evaluation", {}).get("batch_size", 128))
    matched_subset = False
    for subset_label, subset_datasets in table_task_subsets(datasets):
        if task_subset_filter is not None and subset_label != task_subset_filter:
            continue
        matched_subset = True
        if not subset_datasets:
            continue
        pending_datasets = [
            dataset
            for dataset in subset_datasets
            if force
            or not find_evaluation_record(
                config,
                model,
                dataset,
                approach,
                method,
                plan["prompt_run"],
                task_subset=subset_label,
            )
        ]
        if not pending_datasets:
            print(f"[SKIP] {model['id']} {approach} {method} {subset_label}: already evaluated")
            continue

        source_paths = source_paths_for_approach(config, model, subset_datasets, plan, approach)
        existing_sources = [path for path in source_paths if path.is_file()]
        if len(existing_sources) != len(source_paths):
            missing = len(source_paths) - len(existing_sources)
            print(
                f"[SKIP] {model['id']} {approach} {method} {subset_label}: "
                f"missing {missing} DTE checkpoint(s)"
            )
            continue
        if not existing_sources:
            print(f"[SKIP] {model['id']} {approach} {method} {subset_label}: no DTE checkpoints")
            continue

        output_path = budget_merged_path(
            config,
            model,
            approach,
            method,
            plan["prompt_run"],
            task_subset=subset_label,
        )
        if output_path.is_file() and not force:
            payload = torch.load(output_path, map_location="cpu")
        else:
            task_arithmetic_scale = merge_scale_for_subset(config, method, subset_label)
            tsv_m_scale = merge_scale_for_subset(config, "tsv_m", subset_label)
            iso_c_scale = merge_scale_for_subset(config, "iso_c", subset_label)
            payload = merge_state(
                method,
                config,
                model,
                existing_sources,
                device,
                task_arithmetic_scale=task_arithmetic_scale,
                tsv_m_scale=float(tsv_m_scale),
                iso_c_scale=float(iso_c_scale),
            )
            payload.setdefault("metadata", {})
            payload["metadata"].update(
                {
                    "task_subset": subset_label,
                    "subset_datasets": [dataset["name"] for dataset in subset_datasets],
                }
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)

        encoder = vision_encoder_from_payload(config, model, payload, device)
        top1_values = []
        for dataset in pending_datasets:
            prompt_run = prompt_run_for_evaluation(plan, dataset, approach)
            head_path = prompt_head_path(config, model, dataset, prompt_run)
            if not head_path.is_file():
                print(
                    f"[SKIP] {model['id']} {approach} {method} {subset_label} "
                    f"{dataset['name']}: missing {head_path}"
                )
                continue
            head = classification_head_from_payload(torch.load(head_path, map_location="cpu")).to(device)
            metrics = evaluate_classifier(
                config,
                model,
                dataset["eval_dataset"],
                encoder,
                head,
                batch_size,
                data_location=dataset_location(config, dataset),
            )
            record = evaluation_record_payload(
                model,
                dataset,
                approach,
                method,
                plan["prompt_run"],
                prompt_run,
                batch_size,
                metrics,
                task_subset=subset_label,
                merged_checkpoint=str(output_path),
                vision_merge_sources=[str(path) for path in existing_sources],
                source_count=len(existing_sources),
                subset_datasets=[dataset["name"] for dataset in subset_datasets],
            )
            top1_values.append(metrics["top1"])
            upsert_evaluation_record(config, model, record)
            refresh_results_table(config, model, datasets, plan["prompt_run"])
        if top1_values:
            mean_top1 = 100.0 * sum(top1_values) / len(top1_values)
            print(
                f"{model['id']} {approach} {method} {subset_label}: "
                f"mean_top1={mean_top1:.2f}% datasets={len(top1_values)} sources={len(existing_sources)}"
            )
    if task_subset_filter is not None and not matched_subset:
        print(f"[SKIP] {model['id']} {approach} {method} {task_subset_filter}: unknown subset")


def evaluation_task_log(config: Dict[str, Any], model: Dict[str, Any], label: str) -> Path:
    return (
        Path(config["paths"]["logs_root"])
        / safe_token(model["id"])
        / "evaluate"
        / f"{safe_token(label)}.log"
    )


def pre_validation_task_needed(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    plan: Dict[str, Any],
    force: bool,
) -> bool:
    if force:
        return True
    base_prompt_run = plan["prompt_run"]
    if (
        find_evaluation_record(config, model, dataset, "baseline", "vision_ft", base_prompt_run)
        is None
        and vision_checkpoint_path(config, model, dataset, "VisionFT").is_file()
    ):
        return True
    for approach in (APPROACH_EVEN, APPROACH_DATASET):
        prompt_run = prompt_run_for_evaluation(plan, dataset, approach)
        head_path = prompt_head_path(config, model, dataset, prompt_run)
        if (
            find_evaluation_record(
                config, model, dataset, approach, "prompt_learning", base_prompt_run
            )
            is None
            and head_path.is_file()
        ):
            return True
        mode = MODE_EVEN if approach == APPROACH_EVEN else MODE_DATASET
        if (
            find_evaluation_record(config, model, dataset, approach, "dte", base_prompt_run)
            is None
            and head_path.is_file()
            and vision_checkpoint_path(config, model, dataset, mode, base_prompt_run).is_file()
        ):
            return True
    return False


def merged_evaluation_task_needed(
    config: Dict[str, Any],
    model: Dict[str, Any],
    subset_datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    approach: str,
    method: str,
    task_subset: str,
    force: bool,
) -> bool:
    if force:
        return True
    return any(
        find_evaluation_record(
            config,
            model,
            dataset,
            approach,
            method,
            plan["prompt_run"],
            task_subset=task_subset,
        )
        is None
        for dataset in subset_datasets
    )


def evaluation_tasks(
    config: Dict[str, Any],
    source_config_path: Path,
    experiment_id: str,
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    plan: Dict[str, Any],
    run_id: str,
    methods: Sequence[str],
    pre_val: bool,
    force: bool,
    dataset_filter: Optional[str],
) -> List[Task]:
    tasks: List[Task] = []
    script = Path(__file__).resolve()
    base_command = [
        sys.executable,
        str(script),
        "--config",
        str(source_config_path),
        "--stage",
        "evaluate",
        "--experiment-id",
        experiment_id,
        "--model",
        model["id"],
        "--prompt-run",
        run_id,
    ]
    if force:
        base_command.append("--force")

    if pre_val:
        for dataset in datasets:
            if not pre_validation_task_needed(config, model, dataset, plan, force):
                continue
            label = f"EvalPreVal:{model['id']}:{dataset['name']}"
            command = [
                *base_command,
                "--eval-worker",
                "pre_val",
                "--dataset",
                dataset["name"],
            ]
            tasks.append(Task(label, command, evaluation_task_log(config, model, label)))

    for approach in (APPROACH_EVEN, APPROACH_DATASET):
        for method in methods:
            for subset_label, subset_datasets in table_task_subsets(datasets):
                if not subset_datasets or not merged_evaluation_task_needed(
                    config,
                    model,
                    subset_datasets,
                    plan,
                    approach,
                    method,
                    subset_label,
                    force,
                ):
                    continue
                label = f"EvalMerged:{model['id']}:{approach}:{method}:{subset_label}"
                command = [
                    *base_command,
                    "--eval-worker",
                    "merged",
                    "--approach",
                    approach,
                    "--merge-method",
                    method,
                    "--task-subset",
                    subset_label,
                ]
                if dataset_filter is not None:
                    command.extend(["--dataset", dataset_filter])
                tasks.append(Task(label, command, evaluation_task_log(config, model, label)))
    return tasks


def run_evaluation_worker(config: Dict[str, Any], args: argparse.Namespace, run_id: str) -> None:
    if args.model is None:
        raise ValueError("Evaluation workers require --model.")
    model = model_spec(config, args.model)
    datasets = selected_datasets(config, args.dataset)
    plan_path = budget_plan_path(config, model, run_id)
    if not plan_path.is_file():
        raise FileNotFoundError(f"Evaluation worker requires an existing budget plan: {plan_path}")
    plan = read_json(plan_path)

    if args.eval_worker == "pre_val":
        if len(datasets) != 1:
            raise ValueError("pre_val evaluation workers require exactly one --dataset.")
        run_pre_validation(config, model, datasets, plan, args.force)
    elif args.eval_worker == "merged":
        if args.approach is None or args.merge_method is None or args.task_subset is None:
            raise ValueError("merged evaluation workers require --approach, --merge-method, and --task-subset.")
        selected_methods(config, [args.merge_method])
        evaluate_merged_method(
            config,
            model,
            datasets,
            plan,
            args.approach,
            args.merge_method,
            args.force,
            task_subset_filter=args.task_subset,
        )
    else:
        raise ValueError(f"Unknown evaluation worker: {args.eval_worker}")

    refresh_results_table(config, model, config["datasets"], run_id)
    refresh_progress_table(config, model, datasets, run_id)


def run_evaluate_stage(
    config: Dict[str, Any],
    args: argparse.Namespace,
    source_config_path: Path,
    experiment_id: str,
    run_id: str,
    methods: Sequence[str],
    pre_val: bool,
) -> None:
    print("[4/4] Evaluation")
    for model in selected_models(config, args.model):
        datasets = selected_datasets(config, args.dataset)
        plan_path = budget_plan_path(config, model, run_id)
        if plan_path.is_file():
            plan = read_json(plan_path)
        else:
            try:
                plan = build_budget_plan(collect_budget_rows(config, model, datasets, run_id), run_id)
            except FileNotFoundError as error:
                if args.dry_run:
                    print(f"[DRY RUN] evaluate planning skipped: {error}")
                    continue
                raise
            if not args.dry_run:
                write_json(plan_path, plan)
        tasks = evaluation_tasks(
            config,
            source_config_path,
            experiment_id,
            model,
            datasets,
            plan,
            run_id,
            methods,
            pre_val,
            args.force,
            args.dataset,
        )
        print(f"{model['id']} evaluation tasks: {len(tasks)}")
        run_tasks(tasks, config, args.dry_run)
        if not args.dry_run:
            refresh_results_table(config, model, config["datasets"], run_id)
            refresh_progress_table(config, model, datasets, run_id)


TABLE_METHODS = [
    ("vision_ft", "Vision-FT"),
    ("dte", "DTEs"),
    ("prompt_learning", "Prompt learning"),
    ("weight_average", "WA"),
    ("task_arithmetic", "TA"),
    ("tsv_m", "TSV-M"),
    ("iso_c", "Iso-C"),
]

MERGE_TABLE_METHODS = {"weight_average", "task_arithmetic", "tsv_m", "iso_c"}

RESULT_TASK_SUBSETS = [
    (
        "8 Tasks",
        ["EuroSAT", "DTD", "Cars", "SUN397", "SVHN", "RESISC45", "MNIST", "GTSRB"],
    ),
    (
        "14 Tasks",
        [
            "MNIST",
            "GTSRB",
            "EuroSAT",
            "DTD",
            "Cars",
            "FER2013",
            "PCAM",
            "CIFAR100",
            "Flowers102",
            "OxfordIIITPet",
            "STL10",
            "SUN397",
            "SVHN",
            "RESISC45",
        ],
    ),
    (
        "20 Tasks",
        [
            "RESISC45",
            "MNIST",
            "GTSRB",
            "EuroSAT",
            "DTD",
            "Cars",
            "KMNIST",
            "EMNIST",
            "RenderedSST2",
            "FashionMNIST",
            "Food101",
            "CIFAR10",
            "FER2013",
            "PCAM",
            "CIFAR100",
            "Flowers102",
            "OxfordIIITPet",
            "STL10",
            "SUN397",
            "SVHN",
        ],
    ),
]

TASK_ARITHMETIC_SCALE = 0.3
TSV_M_SCALES = {"8 Tasks": 1.0, "14 Tasks": 1.0, "20 Tasks": 0.8}
ISO_C_SCALES = {"8 Tasks": 1.3, "14 Tasks": 1.0, "20 Tasks": 0.9}


def merge_scale_for_subset(
    config: Dict[str, Any],
    method: str,
    task_subset: str,
) -> Optional[float]:
    merge_config = config.get("budget_scheduler", {}).get("merge", {})
    if method == "task_arithmetic":
        value = merge_config.get("task_arithmetic_scale", TASK_ARITHMETIC_SCALE)
        return TASK_ARITHMETIC_SCALE if value is None else float(value)
    if method == "tsv_m":
        configured = merge_config.get("tsv_m_scales", TSV_M_SCALES)
        if not isinstance(configured, dict):
            configured = TSV_M_SCALES
        return float(configured.get(task_subset, TSV_M_SCALES[task_subset]))
    if method == "iso_c":
        configured = merge_config.get("iso_c_scales", ISO_C_SCALES)
        if not isinstance(configured, dict):
            configured = ISO_C_SCALES
        return float(configured.get(task_subset, ISO_C_SCALES[task_subset]))
    return None


def table_task_subsets(datasets: Sequence[Dict[str, Any]]) -> List[tuple[str, List[Dict[str, Any]]]]:
    by_name = {dataset["name"]: dataset for dataset in datasets}
    subsets = []
    for label, names in RESULT_TASK_SUBSETS:
        subsets.append((label, [by_name[name] for name in names if name in by_name]))
    return subsets


def table_record(
    rows: Sequence[Dict[str, Any]],
    dataset: Dict[str, Any],
    approach: str,
    method: str,
    base_prompt_run: str,
    task_subset: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for row in rows:
        if (
            row.get("dataset") == dataset["name"]
            and row.get("approach") == approach
            and row.get("merge_method") == method
            and row.get("base_prompt_run") == base_prompt_run
            and row.get("task_subset") == task_subset
        ):
            return row
    return None


def denominator_record(
    rows: Sequence[Dict[str, Any]],
    dataset: Dict[str, Any],
    approach: str,
    base_prompt_run: str,
) -> Optional[Dict[str, Any]]:
    return table_record(rows, dataset, approach, "dte", base_prompt_run) or table_record(
        rows, dataset, "baseline", "vision_ft", base_prompt_run
    )


def table_cell_values(
    rows: Sequence[Dict[str, Any]],
    datasets: Sequence[Dict[str, Any]],
    task_subset: str,
    approach: str,
    method: str,
    base_prompt_run: str,
) -> Optional[Dict[str, float]]:
    lookup_approach = "baseline" if method == "vision_ft" else approach
    lookup_subset = task_subset if method in MERGE_TABLE_METHODS else None
    method_rows = [
        table_record(rows, dataset, lookup_approach, method, base_prompt_run, lookup_subset)
        for dataset in datasets
    ]
    if any(row is None for row in method_rows):
        return None
    top1_values = [float(row["top1"]) for row in method_rows if row is not None]
    absolute = 100.0 * sum(top1_values) / len(top1_values)
    if method == "prompt_learning":
        return {"absolute": absolute}
    if method in {"vision_ft", "dte"}:
        return {"absolute": absolute, "normalized": 100.0}

    normalized_values = []
    for dataset, row in zip(datasets, method_rows):
        reference = denominator_record(rows, dataset, approach, base_prompt_run)
        if reference is None or float(reference.get("top1", 0.0)) <= 0.0:
            return {"absolute": absolute}
        normalized_values.append(100.0 * float(row["top1"]) / float(reference["top1"]))
    return {
        "absolute": absolute,
        "normalized": sum(normalized_values) / len(normalized_values),
    }


def format_table_cell(values: Optional[Dict[str, float]], bold: bool = False) -> str:
    if values is None:
        return "--"
    absolute = f"{values['absolute']:.2f}"
    if bold:
        absolute = f"\\textbf{{{absolute}}}"
    if "normalized" not in values:
        return absolute
    normalized = f"{values['normalized']:.1f}"
    if bold:
        normalized = f"\\textbf{{{normalized}}}"
    return f"{absolute}$_{{\\scriptstyle({normalized})}}$"


def latex_table_for_approach(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
    approach: str,
    base_prompt_run: str,
) -> str:
    subsets = table_task_subsets(datasets)
    columns = [label for label, _ in subsets]
    values_by_method = {}
    for method, _ in TABLE_METHODS:
        values_by_method[method] = []
        for subset_label, subset in subsets:
            values = table_cell_values(rows, subset, subset_label, approach, method, base_prompt_run)
            values_by_method[method].append(values)

    best_merge_by_column = {}
    for column_index in range(len(subsets)):
        candidates = [
            values_by_method[method][column_index]["absolute"]
            for method in MERGE_TABLE_METHODS
            if values_by_method[method][column_index] is not None
        ]
        best_merge_by_column[column_index] = max(candidates) if candidates else None

    title = "Even Dataset Budget" if approach == APPROACH_EVEN else "Per Dataset Budget"
    label = "even" if approach == APPROACH_EVEN else "per_dataset"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Average Top-1 accuracy (\\%) across vision benchmarks using {model['name']} ({title}). "
        "Subscripts report normalized accuracy relative to the corresponding separate expert, in \\%. "
        "\\textbf{Bold} indicates the best merged score for each setting.}}",
        f"\\label{{tab:budget_{safe_token(model['id'])}_{label}}}",
        "",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{0.55\\linewidth}{!}{",
        "\\begin{tabular}{l " + "c" * len(subsets) + "}",
        "\\toprule",
        "\\textbf{Method} & " + " & ".join(f"\\textbf{{{column}}}" for column in columns) + " \\\\",
        "\\midrule",
    ]
    for method, label_text in TABLE_METHODS:
        if method in {"prompt_learning", "weight_average", "task_arithmetic", "tsv_m", "iso_c"}:
            lines.append("\\midrule")
        cells = []
        for column_index, values in enumerate(values_by_method[method]):
            best = best_merge_by_column[column_index]
            bold = (
                method in MERGE_TABLE_METHODS
                and values is not None
                and best is not None
                and abs(values["absolute"] - best) < 1e-9
            )
            cells.append(format_table_cell(values, bold=bold))
        lines.append(f"{label_text} & " + " & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def refresh_results_table(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    run_id: str,
) -> None:
    rows = read_jsonl(budget_results_path(config, model))
    table_datasets = list(config["datasets"])
    tables = [
        latex_table_for_approach(config, model, table_datasets, rows, APPROACH_EVEN, run_id),
        latex_table_for_approach(config, model, table_datasets, rows, APPROACH_DATASET, run_id),
    ]
    latex = "\n\n".join(tables) + "\n"
    output_dir = Path(config["paths"]["logs_root"]) / safe_token(model["id"])
    text_path = output_dir / "budget_results_table.txt"
    tex_path = output_dir / "budget_results_table.tex"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(text_path, latex)
    atomic_write_text(tex_path, latex)


PROGRESS_COLUMNS = [
    "Dataset",
    "Row",
    "Best Ep",
    "Bwd/Iter",
    "Bwd@Best",
    "Bwd Budget",
    "Time@Best",
    "Best Acc",
]


def metric_cell(metrics: Optional[Dict[str, Any]], key: str, default: str = "---") -> str:
    if not metrics or metrics.get(key) is None:
        return default
    return str(metrics[key])


def progress_epoch(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "---"
    value = metrics.get("epochs_to_best")
    if value is None and metrics.get("best_epoch") is not None:
        value = int(metrics["best_epoch"]) + 1
    return str(value) if value is not None else "---"


def progress_flops(metrics: Optional[Dict[str, Any]], key: str) -> str:
    if not metrics or metrics.get(key) is None:
        return "---"
    return format_flops(float(metrics[key]))


def progress_time(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics or metrics.get("time_to_best_seconds") is None:
        return "---"
    return f"{float(metrics['time_to_best_seconds']):.1f}s"


def progress_accuracy(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics or metrics.get("best_accuracy") is None:
        return "---"
    return f"{100.0 * float(metrics['best_accuracy']):.2f}%"


def progress_row(
    dataset_name: str,
    row_name: str,
    metrics: Optional[Dict[str, Any]],
    budget: Optional[float] = None,
) -> List[str]:
    return [
        dataset_name,
        row_name,
        progress_epoch(metrics),
        progress_flops(metrics, "backward_flops_per_iteration"),
        progress_flops(metrics, "backward_flops_to_best"),
        format_flops(float(budget)) if budget is not None else "---",
        progress_time(metrics),
        progress_accuracy(metrics),
    ]


def progress_latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def progress_table_paths(config: Dict[str, Any], model: Dict[str, Any]) -> tuple[Path, Path]:
    output_dir = Path(config["paths"]["logs_root"]) / safe_token(model["id"])
    return output_dir / "budget_progress_table.md", output_dir / "budget_progress_table.tex"


def progress_plan_lookup(plan: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not plan:
        return {}
    return {row["dataset"]: row for row in plan.get("datasets", [])}


def refresh_progress_table(
    config: Dict[str, Any],
    model: Dict[str, Any],
    datasets: Sequence[Dict[str, Any]],
    run_id: str,
) -> None:
    plan = read_json_if_exists(budget_plan_path(config, model, run_id))
    plan_by_dataset = progress_plan_lookup(plan)
    rows = []
    for dataset in datasets:
        dataset_name = dataset["name"]
        prompt_metrics = read_json_if_exists(prompt_metrics_path(config, model, dataset, run_id))
        vision_metrics = read_json_if_exists(
            vision_metrics_path(vision_checkpoint_path(config, model, dataset, "VisionFT"))
        )
        even_metrics = read_json_if_exists(
            vision_metrics_path(vision_checkpoint_path(config, model, dataset, MODE_EVEN, run_id))
        )
        plan_row = plan_by_dataset.get(dataset_name, {})
        rows.append(progress_row(dataset_name, "Prompt learning", prompt_metrics))
        rows.append(progress_row(dataset_name, "VisionFT", vision_metrics))
        rows.append(
            progress_row(
                dataset_name,
                "DTE (Alg 1)",
                even_metrics,
                budget=plan_row.get("even_budget"),
            )
        )
        if bool(plan_row.get("negative_budget")):
            budget_run = budget_prompt_run_id(run_id)
            alg2_metrics = read_json_if_exists(
                prompt_metrics_path(config, model, dataset, budget_run)
            )
            rows.append(
                progress_row(
                    dataset_name,
                    "Prompt learning (Alg 2)",
                    alg2_metrics,
                    budget=plan_row.get("visionft_backward_to_best"),
                )
            )
        else:
            alg2_metrics = read_json_if_exists(
                vision_metrics_path(
                    vision_checkpoint_path(config, model, dataset, MODE_DATASET, run_id)
                )
            )
            rows.append(
                progress_row(
                    dataset_name,
                    "DTE (Alg 2)",
                    alg2_metrics,
                    budget=plan_row.get("dataset_budget"),
                )
            )

    markdown_lines = [
        "| " + " | ".join(PROGRESS_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(PROGRESS_COLUMNS)) + " |",
    ]
    markdown_lines.extend("| " + " | ".join(row) + " |" for row in rows)

    latex_lines = [
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        " & ".join(f"\\textbf{{{progress_latex_escape(column)}}}" for column in PROGRESS_COLUMNS)
        + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        latex_lines.append(" & ".join(progress_latex_escape(value) for value in row) + " \\\\")
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])

    markdown_path, latex_path = progress_table_paths(config, model)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(markdown_path, "\n".join(markdown_lines) + "\n")
    atomic_write_text(latex_path, "\n".join(latex_lines) + "\n")


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    experiment_id = resolve_experiment_id(config, args)
    apply_experiment_namespace(config, experiment_id)
    print(
        f"Experiment ID: {experiment_id} | "
        f"checkpoints={config['paths']['checkpoints_root']} | "
        f"results={config['paths']['results_root']}"
    )
    bootstrap_prompt_learning(config)
    run_id = selected_prompt_run(config, args.prompt_run)
    if args.eval_worker is not None:
        run_evaluation_worker(config, args, run_id)
        return

    config_path = materialize_experiment_config(config, args.dry_run)
    source_config_path = Path(args.config).expanduser().resolve()
    methods = selected_methods(config, args.methods)
    pre_val = (
        bool(config.get("budget_scheduler", {}).get("pre_val", True))
        if args.pre_val is None
        else bool(args.pre_val)
    )

    if args.stage in {"all", "prompt"}:
        run_prompt_stage(config, config_path, args, run_id)
    if args.stage in {"all", "VisionFT"}:
        run_visionft_stage(config, config_path, args)
    if args.stage in {"all", "budget"}:
        try:
            run_budget_stage(config, config_path, args, run_id)
        except FileNotFoundError as error:
            if args.dry_run and args.stage == "all":
                print(f"[DRY RUN] budget planning skipped: {error}")
                return
            raise
    if args.stage in {"all", "evaluate"}:
        run_evaluate_stage(
            config,
            args,
            source_config_path,
            experiment_id,
            run_id,
            methods,
            pre_val,
        )


if __name__ == "__main__":
    main()
