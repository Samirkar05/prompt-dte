import copy
import fcntl
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open_clip
import torch


OFFICIAL_VALIDATION_ALIASES = {
    "DTDVal",
    "EuroSATVal",
    "Flowers102Val",
    "PCAMVal",
    "RESISC45Val",
    "RenderedSST2Val",
}


def safe_token(value: Any) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in str(value))


def _resolve_path(config_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def load_config(config_path: str) -> Tuple[Dict[str, Any], Path]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    required_paths = [
        "prompt_learning_root",
        "dataset_code_root",
        "data_location",
        "checkpoints_root",
        "results_root",
        "openclip_cache_dir",
        "logs_root",
    ]
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration requires a paths object.")
    for key in required_paths:
        if not paths.get(key):
            raise ValueError(f"Configuration requires paths.{key}.")
        paths[key] = str(_resolve_path(path.parent, paths[key]))

    if not config.get("models"):
        raise ValueError("Configuration requires at least one model.")
    if not config.get("datasets"):
        raise ValueError("Configuration requires at least one dataset.")
    prompt_runs = config.get("prompt_training", {}).get("runs")
    if not prompt_runs:
        raise ValueError("Configuration requires prompt_training.runs.")
    vision_training = config.setdefault("vision_training", {})
    vision_training.setdefault("run_visionft", False)
    if not isinstance(vision_training["run_visionft"], bool):
        raise ValueError("vision_training.run_visionft must be true or false.")

    for model in config["models"]:
        if not isinstance(model, dict) or not model.get("name"):
            raise ValueError("Each model requires a name.")
        model.setdefault("pretrained", "openai")
        model.setdefault("id", safe_token(model["name"]))

    for dataset in config["datasets"]:
        if not isinstance(dataset, dict) or not dataset.get("name"):
            raise ValueError("Each dataset requires a name.")
        dataset.setdefault("train_dataset", f"{dataset['name']}Val")
        dataset.setdefault("eval_dataset", dataset["name"])
        dataset.setdefault(
            "validation_source",
            (
                "official"
                if dataset["train_dataset"] in OFFICIAL_VALIDATION_ALIASES
                else "random_train_split"
            ),
        )
        if dataset.get("data_root"):
            dataset["data_root"] = str(_resolve_path(path.parent, dataset["data_root"]))
        if "prompt_epochs" not in dataset or "vision_epochs" not in dataset:
            raise ValueError(
                f"Dataset {dataset['name']} requires prompt_epochs and vision_epochs."
            )

    for run in prompt_runs:
        if not run.get("id"):
            raise ValueError("Each prompt run requires an id.")
        for key in ("lr", "batch_size", "warmup_length"):
            if key not in run:
                raise ValueError(f"Prompt run {run['id']} requires {key}.")

    model_ids = [model["id"] for model in config["models"]]
    dataset_names = [dataset["name"] for dataset in config["datasets"]]
    run_ids = [run["id"] for run in prompt_runs]
    for label, values in (
        ("model id", model_ids),
        ("dataset name", dataset_names),
        ("prompt run id", run_ids),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Configuration contains duplicate {label} values.")

    return config, path


def bootstrap_prompt_learning(config: Dict[str, Any]) -> Path:
    if bool(config.get("runtime", {}).get("offline_datasets", True)):
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = Path(config["paths"]["prompt_learning_root"])
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"Prompt-learning src directory not found under {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    import src.datasets

    dataset_code_root = Path(config["paths"]["dataset_code_root"])
    if not dataset_code_root.is_dir():
        raise FileNotFoundError(f"Dataset code directory not found: {dataset_code_root}")
    dataset_code_text = str(dataset_code_root)
    if dataset_code_text not in src.datasets.__path__:
        src.datasets.__path__ = [dataset_code_text, *list(src.datasets.__path__)]
    return root


def model_spec(config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for model in config["models"]:
        if model["id"] == model_id or model["name"] == model_id:
            return model
    raise KeyError(f"Unknown model: {model_id}")


def dataset_spec(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    for dataset in config["datasets"]:
        if dataset["name"] == dataset_name or dataset["train_dataset"] == dataset_name:
            return dataset
    raise KeyError(f"Unknown dataset: {dataset_name}")


def dataset_location(config: Dict[str, Any], dataset: Any) -> str:
    specification = dataset if isinstance(dataset, dict) else dataset_spec(config, str(dataset))
    return str(specification.get("data_root") or config["paths"]["data_location"])


def prompt_run_spec(config: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    for run in config["prompt_training"]["runs"]:
        if run["id"] == run_id:
            return run
    raise KeyError(f"Unknown prompt run: {run_id}")


def configured_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def set_seed(config: Dict[str, Any], offset: int = 0) -> None:
    seed = (int(config.get("runtime", {}).get("seed", 0)) + offset) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed_offset(*values: Any) -> int:
    digest = hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def runtime_args(
    config: Dict[str, Any],
    model: Dict[str, Any],
    batch_size: int,
    dataset: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> SimpleNamespace:
    values = {
        "model": model["name"],
        "data_location": (
            dataset_location(config, dataset)
            if dataset is not None
            else config["paths"]["data_location"]
        ),
        "batch_size": batch_size,
        "device": str(configured_device(config)),
        "cache_dir": None,
        "openclip_cachedir": config["paths"]["openclip_cache_dir"],
        "show_eval_progress": bool(config.get("runtime", {}).get("show_progress", True)),
        "results_db": None,
        "eval_datasets": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def create_clip_model(
    config: Dict[str, Any], model: Dict[str, Any]
) -> Tuple[torch.nn.Module, Any, Any]:
    return open_clip.create_model_and_transforms(
        model["name"],
        pretrained=model.get("pretrained", "openai"),
        cache_dir=config["paths"]["openclip_cache_dir"],
    )


class VisionEncoder(torch.nn.Module):
    def __init__(self, visual: torch.nn.Module, train_preprocess: Any, val_preprocess: Any):
        super().__init__()
        self.visual = visual
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.visual(images)
        return output[0] if isinstance(output, tuple) else output


class PromptLearner(torch.nn.Module):
    def __init__(
        self,
        classnames: Sequence[str],
        text_model: torch.nn.Module,
        n_ctx: int,
        device: torch.device,
    ):
        super().__init__()
        context_width = text_model.token_embedding.weight.shape[1]
        context = torch.empty(n_ctx, context_width, dtype=torch.float32)
        torch.nn.init.normal_(context, std=0.02)
        self.ctx = torch.nn.Parameter(context)
        self.n_ctx = n_ctx
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [" ".join(["X"] * n_ctx) + " " + name + "." for name in classnames]
        tokenized_prompts = open_clip.tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = text_model.token_embedding(tokenized_prompts).float()
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.class_count = len(classnames)

    def forward(self) -> torch.Tensor:
        context = self.ctx.unsqueeze(0).expand(self.class_count, -1, -1)
        return torch.cat([self.token_prefix, context, self.token_suffix], dim=1)


class ClassificationHead(torch.nn.Linear):
    def __init__(
        self,
        normalize: bool,
        weights: torch.Tensor,
        biases: Optional[torch.Tensor] = None,
        logit_scale: float = 1.0,
    ):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size, bias=True)
        self.normalize = normalize
        self.logit_scale = float(logit_scale)
        self.weight = torch.nn.Parameter(weights.detach().clone())
        bias = biases if biases is not None else torch.zeros(output_size, dtype=weights.dtype)
        self.bias = torch.nn.Parameter(bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs) * self.logit_scale


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder: VisionEncoder, classification_head: ClassificationHead):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classification_head(self.image_encoder(inputs))


def create_vision_encoder(
    config: Dict[str, Any], model: Dict[str, Any]
) -> Tuple[VisionEncoder, torch.nn.Module]:
    clip_model, train_preprocess, val_preprocess = create_clip_model(config, model)
    vision_encoder = VisionEncoder(clip_model.visual, train_preprocess, val_preprocess)
    return vision_encoder, clip_model


def model_root(config: Dict[str, Any], model: Dict[str, Any]) -> Path:
    return Path(config["paths"]["checkpoints_root"]) / safe_token(model["id"])


def prompt_dir(
    config: Dict[str, Any], model: Dict[str, Any], dataset: Dict[str, Any], run_id: str
) -> Path:
    return model_root(config, model) / "prompts" / safe_token(dataset["name"]) / safe_token(run_id)


def prompt_learner_path(
    config: Dict[str, Any], model: Dict[str, Any], dataset: Dict[str, Any], run_id: str
) -> Path:
    return prompt_dir(config, model, dataset, run_id) / "prompt_learner.pt"


def prompt_head_path(
    config: Dict[str, Any], model: Dict[str, Any], dataset: Dict[str, Any], run_id: str
) -> Path:
    return prompt_dir(config, model, dataset, run_id) / "prompt_head.pt"


def prompt_metrics_path(
    config: Dict[str, Any], model: Dict[str, Any], dataset: Dict[str, Any], run_id: str
) -> Path:
    return prompt_dir(config, model, dataset, run_id) / "metrics.json"


def vision_checkpoint_path(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    mode: str,
    run_id: Optional[str] = None,
) -> Path:
    base = model_root(config, model) / "vision" / mode / safe_token(dataset["name"])
    filename = f"{safe_token(run_id)}.pt" if run_id is not None else "vision_encoder.pt"
    return base / filename


def vision_metrics_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".metrics.json")


def normal_head_path(
    config: Dict[str, Any], model: Dict[str, Any], dataset_name: str
) -> Path:
    return model_root(config, model) / "heads" / f"normal_{safe_token(dataset_name)}.pt"


def merged_vision_path(config: Dict[str, Any], model: Dict[str, Any], run_id: str) -> Path:
    return model_root(config, model) / "merged" / f"DTE_weight_average_{safe_token(run_id)}.pt"


def prompt_results_path(config: Dict[str, Any], model: Dict[str, Any]) -> Path:
    return Path(config["paths"]["results_root"]) / safe_token(model["id"]) / "prompt_runs.jsonl"


def vision_results_path(config: Dict[str, Any], model: Dict[str, Any]) -> Path:
    return Path(config["paths"]["results_root"]) / safe_token(model["id"]) / "vision_runs.jsonl"


def evaluation_results_path(config: Dict[str, Any], model: Dict[str, Any]) -> Path:
    return Path(config["paths"]["results_root"]) / safe_token(model["id"]) / "evaluations.jsonl"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def upsert_jsonl(path: Path, payload: Dict[str, Any], key_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as output_file:
        fcntl.flock(output_file.fileno(), fcntl.LOCK_EX)
        output_file.seek(0)
        rows = []
        for line in output_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(row.get(field) == payload.get(field) for field in key_fields):
                rows.append(row)
        rows.append(payload)
        output_file.seek(0)
        output_file.truncate()
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
        output_file.flush()
        os.fsync(output_file.fileno())
        fcntl.flock(output_file.fileno(), fcntl.LOCK_UN)


def prompt_text_features(clip_model: torch.nn.Module, prompt_learner: torch.nn.Module) -> torch.Tensor:
    prompts = prompt_learner()
    tokenized_prompts = prompt_learner.tokenized_prompts
    hidden = prompts + clip_model.positional_embedding.type(prompts.dtype)
    hidden = hidden.permute(1, 0, 2)
    hidden = clip_model.transformer(hidden, attn_mask=clip_model.attn_mask)
    hidden = hidden.permute(1, 0, 2)
    hidden = clip_model.ln_final(hidden)
    features = hidden[
        torch.arange(hidden.shape[0], device=hidden.device),
        tokenized_prompts.argmax(dim=-1),
    ]
    features = features @ clip_model.text_projection
    return features / features.norm(dim=-1, keepdim=True)


def classification_head_from_payload(payload: Dict[str, Any]) -> torch.nn.Module:
    return ClassificationHead(
        normalize=bool(payload.get("normalize", True)),
        weights=payload["weight"].detach().cpu(),
        biases=payload.get("bias"),
        logit_scale=float(payload["logit_scale"]),
    )


def head_payload(head: torch.nn.Module, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "weight": head.weight.detach().cpu().clone(),
        "bias": head.bias.detach().cpu().clone() if head.bias is not None else None,
        "normalize": bool(getattr(head, "normalize", True)),
        "logit_scale": float(getattr(head, "logit_scale", 1.0)),
    }
    if metadata:
        payload.update(metadata)
    return payload


def build_normal_head(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    clip_model: Optional[torch.nn.Module] = None,
    data_location: Optional[str] = None,
) -> torch.nn.Module:
    path = normal_head_path(config, model, dataset_name)
    if path.is_file():
        return classification_head_from_payload(torch.load(path, map_location="cpu"))

    from src.datasets.registry import get_dataset
    from src.datasets.templates import get_templates

    if clip_model is None:
        clip_model, _, _ = create_clip_model(config, model)
    clip_model = clip_model.to(device).eval()
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location or dataset_location(config, dataset_name),
        batch_size=1,
        num_workers=int(config.get("runtime", {}).get("num_workers", 4)),
    )
    templates = get_templates(dataset_name)
    weights = []
    with torch.no_grad():
        for classname in dataset.classnames:
            texts = open_clip.tokenize([template(classname) for template in templates]).to(device)
            embeddings = clip_model.encode_text(texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embedding = embeddings.mean(dim=0)
            weights.append(embedding / embedding.norm())
    head = ClassificationHead(
        normalize=True,
        weights=torch.stack(weights).cpu(),
        logit_scale=float(clip_model.logit_scale.exp().detach().cpu().item()),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head_payload(head, {"dataset": dataset_name, "model": model["name"]}), path)
    return head


def prompt_head_from_contexts(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset_name: str,
    context_paths: Sequence[Path],
    device: torch.device,
    data_location: Optional[str] = None,
) -> torch.nn.Module:
    from src.datasets.registry import get_dataset

    if not context_paths:
        raise ValueError("At least one prompt context checkpoint is required.")
    checkpoints = [torch.load(path, map_location="cpu") for path in context_paths]
    contexts = [checkpoint["ctx"] for checkpoint in checkpoints]
    reference_shape = contexts[0].shape
    if any(context.shape != reference_shape for context in contexts):
        raise ValueError("Prompt contexts cannot be averaged because their shapes differ.")
    averaged_context = torch.stack([context.float() for context in contexts]).mean(dim=0)

    clip_model, _, _ = create_clip_model(config, model)
    clip_model = clip_model.to(device).eval()
    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location or dataset_location(config, dataset_name),
        batch_size=1,
        num_workers=int(config.get("runtime", {}).get("num_workers", 16)),
    )
    prompt_learner = PromptLearner(
        dataset.classnames,
        clip_model,
        n_ctx=int(reference_shape[0]),
        device=device,
    ).to(device)
    prompt_learner.ctx.data.copy_(averaged_context.to(device, dtype=prompt_learner.ctx.dtype))
    prompt_learner.eval()
    with torch.no_grad():
        weights = prompt_text_features(clip_model, prompt_learner).cpu()
    payload = {
        "weight": weights,
        "bias": None,
        "normalize": True,
        "logit_scale": float(clip_model.logit_scale.exp().detach().cpu().item()),
    }
    return classification_head_from_payload(payload)


def save_vision_checkpoint(
    path: Path,
    vision_encoder: VisionEncoder,
    model: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu().clone() for key, value in vision_encoder.state_dict().items()
    }
    torch.save(
        {
            "model_name": model["name"],
            "model_id": model["id"],
            "pretrained": model.get("pretrained", "openai"),
            "state_dict": state_dict,
            "metadata": metadata,
        },
        path,
    )


def load_vision_checkpoint(
    config: Dict[str, Any], model: Dict[str, Any], path: Path, device: torch.device
) -> VisionEncoder:
    checkpoint = torch.load(path, map_location="cpu")
    encoder, _ = create_vision_encoder(config, model)
    encoder.load_state_dict(checkpoint["state_dict"], strict=True)
    return encoder.to(device)


def average_vision_checkpoints(
    config: Dict[str, Any],
    model: Dict[str, Any],
    paths: Sequence[Path],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if not paths:
        raise ValueError("At least one vision checkpoint is required for averaging.")
    checkpoints = [torch.load(path, map_location="cpu") for path in paths]
    states = [checkpoint["state_dict"] for checkpoint in checkpoints]
    reference_keys = list(states[0].keys())
    if any(list(state.keys()) != reference_keys for state in states[1:]):
        raise ValueError("Vision checkpoints have incompatible state-dict keys.")

    averaged_state = {}
    for key in reference_keys:
        tensors = [state[key] for state in states]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError(f"Vision checkpoint shape mismatch for {key}.")
        if torch.is_floating_point(tensors[0]):
            averaged = torch.stack([tensor.float() for tensor in tensors]).mean(dim=0)
            averaged_state[key] = averaged.to(tensors[0].dtype)
        else:
            averaged_state[key] = tensors[0].clone()

    payload = {
        "model_name": model["name"],
        "model_id": model["id"],
        "pretrained": model.get("pretrained", "openai"),
        "state_dict": averaged_state,
        "metadata": {
            "merge": "equal_weight_average",
            "sources": [str(path) for path in paths],
            "source_count": len(paths),
        },
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
    return payload


def vision_encoder_from_payload(
    config: Dict[str, Any], model: Dict[str, Any], payload: Dict[str, Any], device: torch.device
) -> VisionEncoder:
    encoder, _ = create_vision_encoder(config, model)
    encoder.load_state_dict(payload["state_dict"], strict=True)
    return encoder.to(device)


def evaluate_classifier(
    config: Dict[str, Any],
    model: Dict[str, Any],
    dataset_name: str,
    vision_encoder: VisionEncoder,
    classification_head: torch.nn.Module,
    batch_size: int,
    data_location: Optional[str] = None,
) -> Dict[str, Any]:
    from src.datasets.common import get_dataloader, maybe_dictionarize
    from src.datasets.registry import get_dataset
    device = configured_device(config)
    args = runtime_args(config, model, batch_size, data_location=data_location)
    classifier = ImageClassifier(vision_encoder, classification_head).to(device).eval()
    dataset = get_dataset(
        dataset_name,
        vision_encoder.val_preprocess,
        location=data_location or dataset_location(config, dataset_name),
        batch_size=batch_size,
        num_workers=int(config.get("runtime", {}).get("num_workers", 16)),
    )
    loader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            predictions = classifier(images).argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
    return {"top1": correct / total if total else 0.0, "samples": total}
