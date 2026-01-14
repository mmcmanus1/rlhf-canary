"""Tests for metrics collection."""

import math

import pytest

from canary.collect.metrics import (
    CanaryCallback,
    StabilityMetrics,
    StepTimeStats,
    summarize_step_times,
)


class TestSummarizeStepTimes:
    """Tests for step time summarization."""

    def test_empty_list(self):
        """Test with empty step times."""
        result = summarize_step_times([])
        assert result.count == 0
        assert result.mean is None

    def test_single_step(self):
        """Test with single step time."""
        result = summarize_step_times([0.5])
        assert result.count == 1
        assert result.mean == pytest.approx(0.5)

    def test_warmup_exclusion(self):
        """Test that warmup steps are excluded."""
        # 15 steps, warmup=10 should leave 5
        step_times = [1.0] * 10 + [0.5] * 5
        result = summarize_step_times(step_times, warmup=10)
        assert result.count == 5
        assert result.mean == pytest.approx(0.5)

    def test_warmup_larger_than_data(self):
        """Test when warmup is larger than data."""
        step_times = [0.5, 0.6, 0.7]
        result = summarize_step_times(step_times, warmup=10)
        # Should use all data when warmup > len
        assert result.count == 3

    def test_percentiles(self):
        """Test percentile calculations."""
        # Create sorted data for predictable percentiles
        step_times = [i / 100 for i in range(100)]
        result = summarize_step_times(step_times, warmup=0)

        assert result.count == 100
        assert result.p50 == pytest.approx(0.49)  # Index 49
        assert result.p95 == pytest.approx(0.94)  # Index 94

    def test_statistics(self):
        """Test basic statistics."""
        step_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = summarize_step_times(step_times, warmup=0)

        assert result.mean == pytest.approx(0.3)
        assert result.median == pytest.approx(0.3)
        assert result.min == pytest.approx(0.1)
        assert result.max == pytest.approx(0.5)


class TestCanaryCallback:
    """Tests for CanaryCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        cb = CanaryCallback(warmup_steps=5)
        assert cb.warmup_steps == 5
        assert len(cb.step_times) == 0
        assert cb.nan_steps == 0
        assert cb.max_mem_mb == 0

    def test_step_time_tracking(self):
        """Test that step times are recorded."""
        cb = CanaryCallback()
        cb._step_start_time = 0

        # Simulate a step by manually setting time
        import time

        cb._step_start_time = time.time() - 0.1  # 100ms ago

        class MockArgs:
            pass

        class MockState:
            pass

        class MockControl:
            pass

        cb.on_step_end(MockArgs(), MockState(), MockControl())

        assert len(cb.step_times) == 1
        assert cb.step_times[0] >= 0.1

    def test_nan_detection(self):
        """Test NaN detection in logs."""
        cb = CanaryCallback()

        class MockArgs:
            pass

        class MockState:
            pass

        class MockControl:
            pass

        # Log with NaN
        cb.on_log(MockArgs(), MockState(), MockControl(), logs={"loss": float("nan")})
        assert cb.nan_steps == 1

        # Log with Inf
        cb.on_log(MockArgs(), MockState(), MockControl(), logs={"loss": float("inf")})
        assert cb.inf_steps == 1

        # Normal log
        cb.on_log(MockArgs(), MockState(), MockControl(), logs={"loss": 0.5})
        assert cb.nan_steps == 1  # Unchanged
        assert len(cb.loss_values) == 1

    def test_loss_tracking(self):
        """Test loss value tracking."""
        cb = CanaryCallback()

        class MockArgs:
            pass

        class MockState:
            pass

        class MockControl:
            pass

        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            cb.on_log(MockArgs(), MockState(), MockControl(), logs={"loss": loss})

        assert cb.loss_values == losses

    def test_stability_metrics(self):
        """Test stability metrics generation."""
        cb = CanaryCallback()
        cb.nan_steps = 2
        cb.inf_steps = 1
        cb.loss_values = [1.0, 0.9, 0.8, 0.7, 0.6]

        stability = cb.get_stability_metrics()

        assert stability.nan_steps == 2
        assert stability.inf_steps == 1
        assert stability.final_loss == pytest.approx(0.6)
        assert stability.loss_diverged is False

    def test_loss_divergence_detection(self):
        """Test detection of loss divergence."""
        cb = CanaryCallback()
        # Simulate diverging loss: starts low, ends high
        cb.loss_values = [0.1] * 20 + [0.5] * 20  # Late loss > 1.5x early loss

        stability = cb.get_stability_metrics()
        assert stability.loss_diverged is True

    def test_tokens_per_sec_estimation(self):
        """Test tokens per second estimation."""
        cb = CanaryCallback()
        cb.step_times = [0.5] * 20  # 20 steps at 0.5s each

        tps = cb.estimate_tokens_per_sec(
            batch_size=2,
            gradient_accumulation_steps=4,
            max_length=256,
        )

        # tokens_per_step = 2 * 4 * 256 * 2 (DPO multiplier) = 4096
        # tps = 4096 / 0.5 = 8192
        assert tps == pytest.approx(8192, rel=0.1)
