"""Metrics collection callback for canary training runs."""

from __future__ import annotations

import math
import statistics
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class StepTimeStats(BaseModel):
    """Statistics for step times."""

    count: int
    mean: float | None = None
    median: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None


class PerformanceMetrics(BaseModel):
    """Performance metrics from a canary run."""

    step_time: StepTimeStats
    approx_tokens_per_sec: float | None = None
    max_mem_mb: float = 0.0
    gpu_utilization_avg: float | None = None
    dataloader_wait_pct: float | None = None


class StabilityMetrics(BaseModel):
    """Stability metrics from a canary run."""

    nan_steps: int = 0
    inf_steps: int = 0
    loss_values: list[float] = Field(default_factory=list)
    grad_norm_values: list[float] = Field(default_factory=list)
    final_loss: float | None = None
    loss_diverged: bool = False


class RunConfig(BaseModel):
    """Configuration snapshot for a canary run."""

    model_name: str
    max_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    learning_rate: float
    beta: float | None = None  # DPO-specific
    dataset_name: str
    dataset_size: int
    seed: int | None = None


class CanaryMetrics(BaseModel):
    """Complete metrics from a canary run."""

    run_id: str
    timestamp: str
    duration_seconds: float
    env: Any  # EnvFingerprint (Any to avoid circular import)
    config: RunConfig
    perf: PerformanceMetrics
    stability: StabilityMetrics
    status: str = "completed"  # completed, failed, timeout


def summarize_step_times(step_times: list[float], warmup: int = 10) -> StepTimeStats:
    """Compute statistics from step times, excluding warmup.

    Args:
        step_times: List of step durations in seconds.
        warmup: Number of initial steps to exclude from statistics.

    Returns:
        StepTimeStats with computed statistics.
    """
    # Exclude warmup steps
    times = step_times[warmup:] if len(step_times) > warmup else step_times

    if not times:
        return StepTimeStats(count=0)

    sorted_times = sorted(times)
    n = len(sorted_times)

    def percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return sorted_times[idx]

    return StepTimeStats(
        count=n,
        mean=statistics.mean(times),
        median=statistics.median(times),
        p50=percentile(0.50),
        p95=percentile(0.95),
        p99=percentile(0.99),
        min=min(times),
        max=max(times),
        std=statistics.stdev(times) if n > 1 else 0.0,
    )


class CanaryCallback(TrainerCallback):
    """Callback to collect metrics during training for canary analysis.

    Tracks:
    - Step times (for throughput analysis)
    - Memory usage (peak VRAM)
    - NaN/Inf detection in losses
    - Loss values for stability analysis
    - Gradient norms (when available)
    """

    def __init__(self, warmup_steps: int = 10):
        """Initialize the callback.

        Args:
            warmup_steps: Number of steps to exclude from statistics.
        """
        self.warmup_steps = warmup_steps
        self.step_times: list[float] = []
        self.loss_values: list[float] = []
        self.grad_norm_values: list[float] = []
        self.nan_steps: int = 0
        self.inf_steps: int = 0
        self.max_mem_mb: float = 0.0

        self._step_start_time: float | None = None
        self._run_start_time: float | None = None
        self._tokens_processed: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Reset metrics at the start of training."""
        self._run_start_time = time.time()
        self.step_times = []
        self.loss_values = []
        self.grad_norm_values = []
        self.nan_steps = 0
        self.inf_steps = 0
        self.max_mem_mb = 0.0
        self._tokens_processed = 0

        # Reset CUDA memory stats if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record step start time."""
        self._step_start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record step duration and memory usage."""
        if self._step_start_time is not None:
            step_time = time.time() - self._step_start_time
            self.step_times.append(step_time)

        # Track peak memory
        try:
            import torch

            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                self.max_mem_mb = max(self.max_mem_mb, mem_mb)
        except ImportError:
            pass

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Check logs for NaN/Inf and record loss values."""
        if logs is None:
            return

        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue

            # Check for NaN/Inf
            if math.isnan(value):
                self.nan_steps += 1
            elif math.isinf(value):
                self.inf_steps += 1

            # Track loss values
            if "loss" in key.lower() and not math.isnan(value) and not math.isinf(value):
                self.loss_values.append(value)

            # Track gradient norm
            if "grad_norm" in key.lower() and not math.isnan(value) and not math.isinf(value):
                self.grad_norm_values.append(value)

    def get_duration_seconds(self) -> float:
        """Get total training duration."""
        if self._run_start_time is None:
            return 0.0
        return time.time() - self._run_start_time

    def get_step_time_stats(self) -> StepTimeStats:
        """Get step time statistics."""
        return summarize_step_times(self.step_times, self.warmup_steps)

    def get_stability_metrics(self) -> StabilityMetrics:
        """Get stability metrics from collected data."""
        final_loss = None
        if self.loss_values:
            final_loss = self.loss_values[-1]

        # Detect loss divergence (loss increasing significantly over time)
        loss_diverged = False
        if len(self.loss_values) > 20:
            early_avg = statistics.mean(self.loss_values[:10])
            late_avg = statistics.mean(self.loss_values[-10:])
            # Consider diverged if loss increased by more than 50%
            if late_avg > early_avg * 1.5:
                loss_diverged = True

        return StabilityMetrics(
            nan_steps=self.nan_steps,
            inf_steps=self.inf_steps,
            loss_values=self.loss_values[-100:],  # Keep last 100 for size
            grad_norm_values=self.grad_norm_values[-100:],
            final_loss=final_loss,
            loss_diverged=loss_diverged,
        )

    def estimate_tokens_per_sec(
        self,
        batch_size: int,
        gradient_accumulation_steps: int,
        max_length: int,
        is_dpo: bool = True,
    ) -> float | None:
        """Estimate tokens per second from step times.

        This is an approximation assuming full sequences.

        Args:
            batch_size: Training batch size.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            max_length: Maximum sequence length.
            is_dpo: If True, multiply by 2 for DPO (chosen + rejected sequences).
        """
        step_stats = self.get_step_time_stats()
        if step_stats.mean is None or step_stats.mean == 0:
            return None

        # Approximate tokens per step
        # For DPO: 2 sequences (chosen + rejected) per sample
        # For SFT: 1 sequence per sample
        seq_multiplier = 2 if is_dpo else 1
        tokens_per_step = batch_size * gradient_accumulation_steps * max_length * seq_multiplier
        return tokens_per_step / step_stats.mean
