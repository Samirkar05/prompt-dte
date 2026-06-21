import time
from typing import Callable, Dict

import torch


FLOP_COUNTING_METHOD = (
    "torch.profiler.profile(with_flops=True); counts profiler-supported operators "
    "such as matrix multiplications and 2D convolutions"
)
FLOP_SCOPE = (
    "forward=model(inputs)+loss; backward=loss.backward(); optimizer, scheduler, "
    "data loading, validation, and checkpoint I/O excluded"
)
FLOP_LIMITATIONS = (
    "Lower-bound operator estimate: normalization, softmax, elementwise operations, "
    "and convolution backward may be absent"
)


def format_flops(value):
    if not isinstance(value, (int, float)):
        return "NA"
    magnitude = float(value)
    for divisor, suffix in ((1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M")):
        if abs(magnitude) >= divisor:
            return f"{magnitude / divisor:.3f}{suffix}"
    return f"{magnitude:.0f}"


class TrainingFlopTracker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.profiles: Dict[int, Dict[str, int]] = {}
        self.total_forward_flops = 0
        self.total_backward_flops = 0
        self.total_iterations = 0

    @staticmethod
    def _profile_flops(profiler) -> int:
        return int(
            sum((getattr(event, "flops", 0) or 0) for event in profiler.key_averages())
        )

    def profile_batch(
        self,
        batch_size: int,
        model,
        optimizer,
        loss_closure: Callable[[], torch.Tensor],
        flop_scale: float = 1.0,
    ) -> float:
        if not self.enabled or batch_size in self.profiles:
            return 0.0

        started_at = time.time()
        mutable_buffer_names = ("running_mean", "running_var", "num_batches_tracked")
        buffer_snapshots = [
            (buffer, buffer.detach().clone())
            for name, buffer in model.named_buffers()
            if name.endswith(mutable_buffer_names)
        ]
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            activities = [torch.profiler.ProfilerActivity.CPU]
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
            ) as forward_profiler:
                loss = loss_closure()
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
            ) as backward_profiler:
                loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            forward_flops = int(round(self._profile_flops(forward_profiler) * flop_scale))
            backward_flops = int(round(self._profile_flops(backward_profiler) * flop_scale))
            self.profiles[batch_size] = {
                "forward_flops": forward_flops,
                "backward_flops": backward_flops,
                "flops_per_iteration": forward_flops + backward_flops,
            }
            print(
                "FLOP profile | "
                f"batch_size={batch_size} forward={format_flops(forward_flops)} "
                f"backward={format_flops(backward_flops)} "
                f"iteration={format_flops(forward_flops + backward_flops)}"
            )
        except Exception as exc:
            self.enabled = False
            print(f"WARNING: FLOP profiling disabled after profiler failure: {exc}")
        finally:
            optimizer.zero_grad()
            with torch.no_grad():
                for buffer, saved_buffer in buffer_snapshots:
                    buffer.copy_(saved_buffer)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        return time.time() - started_at

    def record_iteration(self, batch_size: int) -> None:
        if not self.enabled:
            return
        profile = self.profiles.get(batch_size)
        if profile is None:
            return
        self.total_forward_flops += profile["forward_flops"]
        self.total_backward_flops += profile["backward_flops"]
        self.total_iterations += 1

    def profile_for_batch(self, batch_size: int) -> Dict[str, int]:
        return self.profiles.get(
            batch_size,
            {"forward_flops": 0, "backward_flops": 0, "flops_per_iteration": 0},
        )

    def snapshot(self) -> Dict[str, object]:
        total_training_flops = self.total_forward_flops + self.total_backward_flops
        divisor = self.total_iterations if self.total_iterations > 0 else None
        return {
            "flops_per_iteration": total_training_flops / divisor if divisor else None,
            "forward_flops_per_iteration": self.total_forward_flops / divisor if divisor else None,
            "backward_flops_per_iteration": self.total_backward_flops / divisor if divisor else None,
            "total_training_flops": total_training_flops if divisor else None,
            "total_forward_flops": self.total_forward_flops if divisor else None,
            "total_backward_flops": self.total_backward_flops if divisor else None,
            "flop_profiled_batch_sizes": {
                str(batch_size): profile for batch_size, profile in sorted(self.profiles.items())
            },
            "flop_counting_method": FLOP_COUNTING_METHOD,
            "flop_scope": FLOP_SCOPE,
            "flop_limitations": FLOP_LIMITATIONS,
            "flop_estimate_is_lower_bound": True,
        }
