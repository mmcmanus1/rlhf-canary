"""Metrics collection callback for canary training runs."""

from __future__ import annotations

import logging
import math
import statistics
import time
from typing import Any

from pydantic import BaseModel, Field
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

# Keys to check for NaN/Inf in stability metrics
# Only these keys trigger nan_steps/inf_steps increments
STABILITY_KEYS = {
    "loss",
    "train_loss",
    "policy_loss",
    "value_loss",
    "grad_norm",
    "rewards/chosen",
    "rewards/rejected",
    "kl",
}


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
    profiler: Any | None = None  # ProfilerSummary (Any to avoid circular import)
    status: str = "completed"  # completed, failed, timeout


def summarize_step_times(step_times: list[float], warmup: int = 10) -> StepTimeStats:
    """Compute statistics from step times, excluding warmup.

    Args:
        step_times: List of step durations in seconds.
        warmup: Number of initial steps to exclude from statistics.

    Returns:
        StepTimeStats with computed statistics.
    """
    # Exclude warmup steps, with warning if warmup exceeds total
    if len(step_times) <= warmup:
        if step_times:
            logger.warning(
                f"Warmup steps ({warmup}) >= total steps ({len(step_times)}). "
                "Using all steps for statistics. Consider reducing warmup or increasing max_steps."
            )
        times = step_times
    else:
        times = step_times[warmup:]

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
    - GPU utilization (requires pynvml)
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

        # GPU utilization tracking
        self._gpu_utilizations: list[float] = []
        self._nvml_available: bool = False
        self._nvml_handle: Any = None

        # Dataloader wait tracking
        self._dataloader_wait_times: list[float] = []
        self._step_end_time: float | None = None

        self._step_start_time: float | None = None
        self._run_start_time: float | None = None
        self._tokens_processed: int = 0

        # Try to initialize NVML for GPU utilization monitoring
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVML for GPU utilization monitoring (optional)."""
        try:
            import pynvml

            pynvml.nvmlInit()
            # Get handle for first GPU (device 0)
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            logger.debug("NVML initialized for GPU utilization monitoring")
        except ImportError:
            logger.debug("pynvml not installed, GPU utilization monitoring disabled")
        except Exception as e:
            logger.debug(f"Failed to initialize NVML: {e}")

    def _sample_gpu_utilization(self) -> None:
        """Sample current GPU utilization if NVML is available."""
        if not self._nvml_available or self._nvml_handle is None:
            return

        try:
            import pynvml

            util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            self._gpu_utilizations.append(float(util.gpu))
        except Exception:
            # Don't fail training if GPU sampling fails
            pass

    def get_gpu_utilization_avg(self) -> float | None:
        """Get average GPU utilization across all samples.

        Returns:
            Average GPU utilization percentage (0-100), or None if no data.
        """
        if not self._gpu_utilizations:
            return None
        return statistics.mean(self._gpu_utilizations)

    def get_dataloader_wait_pct(self) -> float | None:
        """Get dataloader wait time as percentage of total step time.

        This measures the fraction of time spent waiting for the next batch
        from the dataloader, which can indicate CPU/IO bottlenecks.

        Returns:
            Percentage of step time spent waiting for data (0-100), or None if no data.
        """
        if not self._dataloader_wait_times or not self.step_times:
            return None

        # Use matching lengths (we have one fewer wait time than step times)
        n = min(len(self._dataloader_wait_times), len(self.step_times) - 1)
        if n == 0:
            return None

        total_wait = sum(self._dataloader_wait_times[:n])
        total_step = sum(self.step_times[1 : n + 1])  # Skip first step (no wait time)

        if total_step == 0:
            return None

        return 100.0 * total_wait / total_step

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
        self._gpu_utilizations = []
        self._dataloader_wait_times = []
        self._step_end_time = None

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
        """Record step start time and dataloader wait time."""
        current_time = time.time()

        # Track dataloader wait time (time between previous step end and this step begin)
        # This approximates time spent fetching the next batch from the dataloader
        if self._step_end_time is not None:
            wait_time = current_time - self._step_end_time
            self._dataloader_wait_times.append(wait_time)

        self._step_start_time = current_time

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Record step duration, memory usage, and GPU utilization."""
        current_time = time.time()

        if self._step_start_time is not None:
            step_time = current_time - self._step_start_time
            self.step_times.append(step_time)

        # Record step end time for dataloader wait calculation
        self._step_end_time = current_time

        # Track peak memory
        try:
            import torch

            if torch.cuda.is_available():
                mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                self.max_mem_mb = max(self.max_mem_mb, mem_mb)
        except ImportError:
            pass

        # Sample GPU utilization
        self._sample_gpu_utilization()

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

        # Track whether we found NaN/Inf in this log step (count once per step, not per key)
        found_nan = False
        found_inf = False

        for key, value in logs.items():
            if not isinstance(value, (int, float)):
                continue

            # Check for NaN/Inf only in stability-relevant keys
            # This avoids false positives from learning_rate, epoch, etc.
            key_lower = key.lower()
            is_stability_key = any(sk in key_lower for sk in STABILITY_KEYS)

            if is_stability_key:
                if math.isnan(value):
                    found_nan = True
                elif math.isinf(value):
                    found_inf = True

            # Track loss values
            if "loss" in key_lower and not math.isnan(value) and not math.isinf(value):
                self.loss_values.append(value)

            # Track gradient norm
            if "grad_norm" in key_lower and not math.isnan(value) and not math.isinf(value):
                self.grad_norm_values.append(value)

        # Increment counters once per step, not per key
        if found_nan:
            self.nan_steps += 1
        if found_inf:
            self.inf_steps += 1

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

        NOTE: This is an upper-bound approximation that assumes all sequences
        are padded to max_length. Real throughput may be 30-50% lower due to
        variable sequence lengths in actual datasets. Use for relative comparisons
        between runs with the same config, not absolute throughput claims.

        Args:
            batch_size: Training batch size.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            max_length: Maximum sequence length.
            is_dpo: If True, multiply by 2 for DPO (chosen + rejected sequences).

        Returns:
            Approximate tokens per second, or None if insufficient data.
        """
        step_stats = self.get_step_time_stats()
        if step_stats.mean is None or step_stats.mean == 0:
            return None

        # Approximate tokens per step (upper bound - assumes full sequences)
        # For DPO: 2 sequences (chosen + rejected) per sample
        # For SFT: 1 sequence per sample
        seq_multiplier = 2 if is_dpo else 1
        tokens_per_step = batch_size * gradient_accumulation_steps * max_length * seq_multiplier

        tps = tokens_per_step / step_stats.mean

        logger.debug(
            f"Tokens/sec estimate: {tps:.0f} (upper bound, assumes max_length={max_length})"
        )

        return tps
