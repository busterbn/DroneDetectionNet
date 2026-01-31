"""Simple PyTorch profiling utilities."""

from pathlib import Path

import torch

from drone_detector_mlops.utils.logger import get_logger

logger = get_logger(__name__)


class ProfilerWithTable:
    """Wrapper around torch.profiler that prints a table summary."""

    def __init__(self, profiler, print_table: bool = True):
        self.profiler = profiler
        self.print_table = print_table

    def __enter__(self):
        return self.profiler.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self.profiler.__exit__(exc_type, exc_val, exc_tb)
        if self.print_table:
            self._print_summary()
        return result

    def step(self):
        """Step the profiler forward (call after each training batch)."""
        self.profiler.step()

    def _print_summary(self):
        """Log profiler summary tables for CPU time and memory usage."""
        key_averages = self.profiler.key_averages()

        cpu_table = key_averages.table(sort_by="cpu_time_total", row_limit=15)
        logger.info("PyTorch Profiler Summary (CPU time):\n" + cpu_table)

        memory_table = key_averages.table(sort_by="self_cpu_memory_usage", row_limit=10)
        logger.info("Top operations by memory:\n" + memory_table)


def get_profiler(output_dir: str = "profiler", print_table: bool = True):
    """
    Create a PyTorch profiler for training.

    The profiler output will be visible in Cloud Logs when running on Vertex AI.

    Args:
        output_dir: Directory to save profiler traces (for local TensorBoard viewing)
        print_table: Whether to print summary table after profiling

    Returns:
        ProfilerWithTable: Context manager that wraps torch.profiler.profile
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect hardware
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        logger.info("GPU detected - enabling CUDA profiling")

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,  # Skip first step (warmup)
            warmup=1,  # Warmup for 1 step
            active=3,  # Record 3 steps
            repeat=1,  # One cycle only
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    return ProfilerWithTable(profiler=profiler, print_table=print_table)
