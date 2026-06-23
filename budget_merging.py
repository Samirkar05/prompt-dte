from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from pipeline_utils import create_vision_encoder


@dataclass
class BudgetTaskVector:
    vector: Dict[str, torch.Tensor]


def _load_vision_state(path: Any) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"Vision checkpoint does not contain a state_dict: {path}")
    return checkpoint["state_dict"]


def base_vision_state(config: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    encoder, clip_model = create_vision_encoder(config, model)
    del clip_model
    return {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}


def task_vectors_from_checkpoints(
    base_state: Dict[str, torch.Tensor],
    paths: Sequence[Any],
) -> Tuple[Sequence[BudgetTaskVector], Sequence[Dict[str, torch.Tensor]]]:
    finetuned_states = [_load_vision_state(path) for path in paths]
    reference_keys = list(base_state.keys())
    vectors = []
    for path, state in zip(paths, finetuned_states):
        if list(state.keys()) != reference_keys:
            raise ValueError(f"State-dict keys do not match base encoder: {path}")
        vector = {}
        for key in reference_keys:
            base_tensor = base_state[key]
            tensor = state[key]
            if tensor.shape != base_tensor.shape:
                raise ValueError(f"Shape mismatch for {key} in {path}")
            if torch.is_floating_point(base_tensor):
                vector[key] = tensor.float() - base_tensor.float()
        vectors.append(BudgetTaskVector(vector=vector))
    return vectors, finetuned_states


def _state_from_vector(
    base_state: Dict[str, torch.Tensor],
    vector: Dict[str, torch.Tensor],
    scaling: float,
) -> Dict[str, torch.Tensor]:
    merged = {}
    for key, base_tensor in base_state.items():
        if key not in vector:
            merged[key] = base_tensor.detach().cpu().clone()
            continue
        updated = base_tensor.float() + float(scaling) * vector[key].float().cpu()
        merged[key] = updated.to(dtype=base_tensor.dtype)
    return merged


def weight_average_state(
    base_state: Dict[str, torch.Tensor],
    finetuned_states: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if not finetuned_states:
        raise ValueError("At least one finetuned state is required.")
    merged = {}
    for key, base_tensor in base_state.items():
        tensors = [state[key] for state in finetuned_states]
        if torch.is_floating_point(base_tensor):
            merged[key] = torch.stack([tensor.float() for tensor in tensors]).mean(dim=0).to(
                dtype=base_tensor.dtype
            )
        else:
            merged[key] = tensors[0].detach().cpu().clone()
    return merged


def task_arithmetic_state(
    base_state: Dict[str, torch.Tensor],
    task_vectors: Sequence[BudgetTaskVector],
    scaling: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    if not task_vectors:
        raise ValueError("At least one task vector is required.")
    scale = (1.0 / len(task_vectors)) if scaling is None else float(scaling)
    summed = {}
    for key in task_vectors[0].vector:
        summed[key] = sum(task_vector.vector[key].float() for task_vector in task_vectors)
    return _state_from_vector(base_state, summed, scale)


def _resolve_sv_reduction_weights(task_vectors, sv_reduction=None):
    num_vectors = len(task_vectors)
    if num_vectors == 0:
        raise ValueError("task_vectors must contain at least one vector.")
    if sv_reduction is None:
        return [1.0 / num_vectors] * num_vectors
    if len(sv_reduction) != num_vectors:
        raise ValueError("sv_reduction must have the same length as task_vectors.")
    total = float(sum(sv_reduction))
    if total <= 0:
        raise ValueError("sv_reduction must sum to a positive value.")
    return [float(weight) / total for weight in sv_reduction]


def _allocate_rank_by_weights(rank: int, weights: Sequence[float]) -> Sequence[int]:
    raw_counts = [rank * weight for weight in weights]
    counts = [int(value) for value in raw_counts]
    remaining = rank - sum(counts)
    fractions = sorted(
        ((raw_counts[index] - counts[index], index) for index in range(len(weights))),
        reverse=True,
    )
    for _, index in fractions[:remaining]:
        counts[index] += 1
    return counts


def tsv_m_vector(task_vectors, config, sv_reduction=None):
    """
    Copied/adapted from model-merging/main_TSV.py::tsv_wm.
    Merges each matrix task tensor by assigning equal low-rank SVD slices to tasks.
    """
    sv_reduction_weights = _resolve_sv_reduction_weights(task_vectors, sv_reduction)
    device = config.device
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            rank_slices = None
            for index, task_vector in enumerate(task_vectors):
                vec = task_vector.vector[key].to(device)
                if len(task_vector.vector[key].shape) == 2 and "text_projection" not in key:
                    u, s, v = torch.linalg.svd(vec, full_matrices=False)
                    if index == 0:
                        sum_u = torch.zeros_like(u, device=device)
                        sum_s = torch.zeros_like(s, device=device)
                        sum_v = torch.zeros_like(v, device=device)
                        rank_counts = _allocate_rank_by_weights(s.shape[0], sv_reduction_weights)
                        rank_slices = []
                        start = 0
                        for count in rank_counts:
                            stop = start + count
                            rank_slices.append(slice(start, stop))
                            start = stop
                    current_slice = rank_slices[index]
                    reduced_rank = current_slice.stop - current_slice.start
                    if reduced_rank > 0:
                        sum_u[:, current_slice] = u[:, :reduced_rank]
                        sum_s[current_slice] = s[:reduced_rank]
                        sum_v[current_slice, :] = v[:reduced_rank, :]
                else:
                    if index == 0:
                        new_vector[key] = vec.clone()
                    else:
                        new_vector[key] += (vec - new_vector[key]) / (index + 1)

            reference = task_vectors[0].vector[key]
            if len(reference.shape) == 2 and "text_projection" not in key:
                u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
                u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
                new_vector[key] = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
    return {key: value.detach().cpu() for key, value in new_vector.items()}


def iso_c_vector(task_vectors, config):
    """
    Copied/adapted from model-merging/main_IsoC.py::iso_c.
    Averages task tensors, then equalizes singular values for matrix parameters.
    """
    device = config.device
    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            tensors = [task_vector.vector[key].to(device) for task_vector in task_vectors]
            new_vector[key] = sum(tensors) / len(tensors)
            if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
                new_vector[key] *= len(tensors)
                u, s, v = torch.linalg.svd(new_vector[key], full_matrices=False)
                s_mean = torch.ones_like(s) * s.mean()
                new_vector[key] = torch.linalg.multi_dot((u, torch.diag(s_mean), v))
    return {key: value.detach().cpu() for key, value in new_vector.items()}


def merge_state(
    method: str,
    config: Dict[str, Any],
    model: Dict[str, Any],
    source_paths: Sequence[Any],
    device: torch.device,
    task_arithmetic_scale: Optional[float] = None,
    tsv_m_scale: float = 1.0,
    iso_c_scale: float = 1.0,
) -> Dict[str, Any]:
    base_state = base_vision_state(config, model)
    task_vectors, finetuned_states = task_vectors_from_checkpoints(base_state, source_paths)
    merge_config = SimpleNamespace(device=device, eval_datasets=[str(path) for path in source_paths])

    if method == "weight_average":
        merged_state = weight_average_state(base_state, finetuned_states)
        scaling = None
    elif method == "task_arithmetic":
        merged_state = task_arithmetic_state(base_state, task_vectors, task_arithmetic_scale)
        scaling = (1.0 / len(task_vectors)) if task_arithmetic_scale is None else task_arithmetic_scale
    elif method == "tsv_m":
        vector = tsv_m_vector(task_vectors, merge_config)
        merged_state = _state_from_vector(base_state, vector, tsv_m_scale)
        scaling = tsv_m_scale
    elif method == "iso_c":
        vector = iso_c_vector(task_vectors, merge_config)
        merged_state = _state_from_vector(base_state, vector, iso_c_scale)
        scaling = iso_c_scale
    else:
        raise ValueError(f"Unknown merge method: {method}")

    return {
        "model_name": model["name"],
        "model_id": model["id"],
        "pretrained": model.get("pretrained", "openai"),
        "state_dict": merged_state,
        "metadata": {
            "merge": method,
            "sources": [str(path) for path in source_paths],
            "source_count": len(source_paths),
            "scaling": scaling,
        },
    }
