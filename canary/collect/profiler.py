"""PyTorch profiler integration for detailed performance analysis."""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Generator

from pydantic import BaseModel
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class ProfilerConfig(BaseModel):
    """Configuration for profiler capture."""

    enabled: bool = False
    start_step: int = 50
    num_steps: int = 20
    output_dir: str = "./profiler_traces"
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False  # Can be slow


class ProfilerSummary(BaseModel):
    """Summary of profiler results."""

    trace_path: str | None = None
    cuda_time_total_ms: float = 0.0
    cpu_time_total_ms: float = 0.0
    self_cuda_time_total_ms: float = 0.0
    top_cuda_ops: list[dict[str, Any]] = []
    top_cpu_ops: list[dict[str, Any]] = []


@contextmanager
def profiler_context(
    config: ProfilerConfig,
    step: int,
    run_id: str,
) -> Generator[Any, None, None]:
    """Context manager for optional profiling during training steps.

    Args:
        config: Profiler configuration.
        step: Current training step.
        run_id: Run identifier for trace naming.

    Yields:
        PyTorch profiler context or None if profiling not active.
    """
    if not config.enabled:
        yield None
        return

    # Only profile within the configured window
    if step < config.start_step or step >= config.start_step + config.num_steps:
        yield None
        return

    try:
        import torch
        from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        # Create output directory
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_dir = output_dir / run_id
        trace_dir.mkdir(parents=True, exist_ok=True)

        with profile(
            activities=activities,
            record_shapes=config.record_shapes,
            profile_memory=config.profile_memory,
            with_stack=config.with_stack,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        ) as prof:
            yield prof

    except ImportError:
        yield None


class ProfilerCallback:
    """Callback for capturing profiler traces during specific training windows.

    This integrates with the TrainerCallback but manages the profiler separately
    since PyTorch profiler has its own context management requirements.
    """

    def __init__(self, config: ProfilerConfig, run_id: str):
        """Initialize profiler callback.

        Args:
            config: Profiler configuration.
            run_id: Run identifier for trace naming.
        """
        self.config = config
        self.run_id = run_id
        self._profiler = None
        self._exit_stack: ExitStack | None = None
        self._summary: ProfilerSummary | None = None

    def should_profile(self, step: int) -> bool:
        """Check if profiling should be active for this step."""
        if not self.config.enabled:
            return False
        return self.config.start_step <= step < self.config.start_step + self.config.num_steps

    def start_profiling(self) -> None:
        """Start profiling session."""
        if not self.config.enabled:
            return

        try:
            import torch
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            output_dir = Path(self.config.output_dir) / self.run_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Use ExitStack for proper cleanup on exceptions
            self._exit_stack = ExitStack()
            profiler = profile(
                activities=activities,
                record_shapes=self.config.record_shapes,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack,
            )
            self._profiler = self._exit_stack.enter_context(profiler)

        except ImportError:
            pass

    def step(self) -> None:
        """Signal profiler to record a step."""
        if self._profiler is not None:
            self._profiler.step()

    def stop_profiling(self) -> ProfilerSummary:
        """Stop profiling and generate summary."""
        if self._profiler is None:
            return ProfilerSummary()

        # Generate summary before closing (profiler data needed)
        summary = self._generate_summary()

        # Properly close the profiler context via ExitStack
        if self._exit_stack is not None:
            self._exit_stack.close()
            self._exit_stack = None

        self._profiler = None
        return summary

    def _generate_summary(self) -> ProfilerSummary:
        """Generate summary from profiler data."""
        if self._profiler is None:
            return ProfilerSummary()

        try:
            # Get key averages
            key_averages = self._profiler.key_averages()

            # Export trace
            output_dir = Path(self.config.output_dir) / self.run_id
            trace_path = output_dir / "trace.json"
            self._profiler.export_chrome_trace(str(trace_path))

            # Compute totals
            cuda_time_total = sum(
                item.cuda_time_total for item in key_averages if item.cuda_time_total
            )
            cpu_time_total = sum(
                item.cpu_time_total for item in key_averages if item.cpu_time_total
            )
            self_cuda_time_total = sum(
                item.self_cuda_time_total for item in key_averages if item.self_cuda_time_total
            )

            # Get top ops by CUDA time
            cuda_sorted = sorted(
                key_averages,
                key=lambda x: x.self_cuda_time_total or 0,
                reverse=True,
            )
            top_cuda_ops = [
                {
                    "name": item.key,
                    "self_cuda_time_ms": (item.self_cuda_time_total or 0) / 1000,
                    "count": item.count,
                }
                for item in cuda_sorted[:10]
            ]

            # Get top ops by CPU time
            cpu_sorted = sorted(
                key_averages,
                key=lambda x: x.self_cpu_time_total or 0,
                reverse=True,
            )
            top_cpu_ops = [
                {
                    "name": item.key,
                    "self_cpu_time_ms": (item.self_cpu_time_total or 0) / 1000,
                    "count": item.count,
                }
                for item in cpu_sorted[:10]
            ]

            return ProfilerSummary(
                trace_path=str(trace_path) if trace_path.exists() else None,
                cuda_time_total_ms=cuda_time_total / 1000,
                cpu_time_total_ms=cpu_time_total / 1000,
                self_cuda_time_total_ms=self_cuda_time_total / 1000,
                top_cuda_ops=top_cuda_ops,
                top_cpu_ops=top_cpu_ops,
            )

        except Exception:
            return ProfilerSummary()


class ProfilerTrainerCallback(TrainerCallback):
    """TrainerCallback that manages profiler lifecycle during training.

    This integrates the ProfilerCallback with HuggingFace's Trainer,
    automatically starting/stopping profiling at the configured steps.
    """

    def __init__(self, config: ProfilerConfig, run_id: str):
        """Initialize the profiler trainer callback.

        Args:
            config: Profiler configuration.
            run_id: Run identifier for trace naming.
        """
        self.config = config
        self.run_id = run_id
        self._profiler_callback = ProfilerCallback(config, run_id)
        self._profiling_started = False
        self._profiling_stopped = False
        self._current_step = 0
        self.summary: ProfilerSummary | None = None

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Check if we should start profiling on this step."""
        self._current_step = state.global_step

        # Start profiling when we reach the start step
        if (
            self.config.enabled
            and not self._profiling_started
            and not self._profiling_stopped
            and state.global_step >= self.config.start_step
        ):
            logger.info(
                f"Starting profiler at step {state.global_step} "
                f"(will profile {self.config.num_steps} steps)"
            )
            self._profiler_callback.start_profiling()
            self._profiling_started = True

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Step the profiler and check if we should stop."""
        if self._profiling_started and not self._profiling_stopped:
            self._profiler_callback.step()

            # Stop profiling when we've captured enough steps
            end_step = self.config.start_step + self.config.num_steps
            if state.global_step >= end_step:
                logger.info(f"Stopping profiler at step {state.global_step}")
                self.summary = self._profiler_callback.stop_profiling()
                self._profiling_stopped = True
                if self.summary.trace_path:
                    logger.info(f"Profiler trace saved to: {self.summary.trace_path}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Ensure profiler is stopped on training end."""
        if self._profiling_started and not self._profiling_stopped:
            logger.info("Stopping profiler at training end")
            self.summary = self._profiler_callback.stop_profiling()
            self._profiling_stopped = True

    def get_summary(self) -> ProfilerSummary | None:
        """Get the profiler summary if profiling was performed."""
        return self.summary
