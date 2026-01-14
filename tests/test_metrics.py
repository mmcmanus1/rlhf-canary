"""Tests for metrics collection."""

import math
import time

import pytest

from canary.collect.metrics import (
    CanaryCallback,
    STABILITY_KEYS,
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

    def test_loss_divergence_with_negative_losses(self):
        """Test loss divergence detection with negative losses (e.g., PPO policy loss)."""
        cb = CanaryCallback()
        # Negative losses: more negative means divergence (larger magnitude)
        # early_avg = -0.1, late_avg = -0.5 -> diverged (magnitude increased)
        cb.loss_values = [-0.1] * 20 + [-0.5] * 20

        stability = cb.get_stability_metrics()
        assert stability.loss_diverged is True

    def test_loss_convergence_with_negative_losses(self):
        """Test that improving negative losses are not flagged as divergence."""
        cb = CanaryCallback()
        # Negative losses improving: less negative over time (smaller magnitude)
        # early_avg = -0.5, late_avg = -0.2 -> NOT diverged (magnitude decreased)
        cb.loss_values = [-0.5] * 20 + [-0.2] * 20

        stability = cb.get_stability_metrics()
        assert stability.loss_diverged is False

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


class TestPhaseAFixes:
    """Tests for Phase A bug fixes."""

    def test_nan_detection_only_stability_keys(self):
        """Test that NaN/Inf detection only triggers for stability-relevant keys.

        Phase A fix: NaN in learning_rate should NOT increment nan_steps,
        but NaN in loss should.
        """
        cb = CanaryCallback()

        class Mock:
            pass

        # NaN in non-stability key should NOT count
        cb.on_log(Mock(), Mock(), Mock(), logs={"learning_rate": float("nan")})
        assert cb.nan_steps == 0, "NaN in learning_rate should not trigger nan_steps"

        cb.on_log(Mock(), Mock(), Mock(), logs={"epoch": float("nan")})
        assert cb.nan_steps == 0, "NaN in epoch should not trigger nan_steps"

        # NaN in stability key SHOULD count
        cb.on_log(Mock(), Mock(), Mock(), logs={"loss": float("nan")})
        assert cb.nan_steps == 1, "NaN in loss should trigger nan_steps"

        cb.on_log(Mock(), Mock(), Mock(), logs={"train_loss": float("nan")})
        assert cb.nan_steps == 2, "NaN in train_loss should trigger nan_steps"

        cb.on_log(Mock(), Mock(), Mock(), logs={"grad_norm": float("nan")})
        assert cb.nan_steps == 3, "NaN in grad_norm should trigger nan_steps"

    def test_inf_detection_only_stability_keys(self):
        """Test that Inf detection only triggers for stability-relevant keys."""
        cb = CanaryCallback()

        class Mock:
            pass

        # Inf in non-stability key should NOT count
        cb.on_log(Mock(), Mock(), Mock(), logs={"learning_rate": float("inf")})
        assert cb.inf_steps == 0, "Inf in learning_rate should not trigger inf_steps"

        # Inf in stability key SHOULD count
        cb.on_log(Mock(), Mock(), Mock(), logs={"loss": float("inf")})
        assert cb.inf_steps == 1, "Inf in loss should trigger inf_steps"

    def test_nan_inf_counted_once_per_step(self):
        """Test that multiple NaN/Inf in same log step only increment once.

        Phase A fix: If both loss and grad_norm are NaN in the same on_log call,
        nan_steps should only increment by 1.
        """
        cb = CanaryCallback()

        class Mock:
            pass

        # Multiple NaN values in same log step
        cb.on_log(
            Mock(),
            Mock(),
            Mock(),
            logs={
                "loss": float("nan"),
                "train_loss": float("nan"),
                "grad_norm": float("nan"),
            },
        )
        assert cb.nan_steps == 1, "Multiple NaNs in same step should only count once"

    def test_stability_keys_comprehensive(self):
        """Test that all documented stability keys are checked."""
        expected_keys = {"loss", "train_loss", "policy_loss", "value_loss", "grad_norm", "kl"}
        # STABILITY_KEYS should contain at least these
        for key in expected_keys:
            assert key in STABILITY_KEYS, f"Expected '{key}' in STABILITY_KEYS"

    def test_warmup_warning_logged(self, caplog):
        """Test that a warning is logged when warmup >= total steps.

        Phase A fix: Should warn when warmup is larger than data.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            result = summarize_step_times([0.1, 0.2, 0.3], warmup=10)

        # Should still return stats (using all data)
        assert result.count == 3
        # Should have logged a warning
        assert any("Warmup steps" in record.message for record in caplog.records)


class TestPhaseBFeatures:
    """Tests for Phase B new features (GPU utilization, dataloader wait)."""

    def test_gpu_utilization_avg_no_data(self):
        """Test GPU utilization returns None when no data collected."""
        cb = CanaryCallback()
        assert cb.get_gpu_utilization_avg() is None

    def test_gpu_utilization_avg_with_data(self):
        """Test GPU utilization average calculation."""
        cb = CanaryCallback()
        # Manually add some utilization samples
        cb._gpu_utilizations = [80.0, 90.0, 85.0, 95.0]

        avg = cb.get_gpu_utilization_avg()
        assert avg == pytest.approx(87.5)

    def test_dataloader_wait_pct_no_data(self):
        """Test dataloader wait returns None when no data collected."""
        cb = CanaryCallback()
        assert cb.get_dataloader_wait_pct() is None

    def test_dataloader_wait_pct_calculation(self):
        """Test dataloader wait percentage calculation."""
        cb = CanaryCallback()

        class Mock:
            pass

        # Simulate 5 steps
        # Step 1: no wait time (first step)
        cb._step_end_time = None
        cb.on_step_begin(Mock(), Mock(), Mock())
        time.sleep(0.01)  # Simulate step work
        cb._step_start_time = time.time() - 0.1  # Pretend step took 0.1s
        cb.on_step_end(Mock(), Mock(), Mock())

        # Steps 2-5: have wait times
        for _ in range(4):
            # Simulate dataloader wait of ~0.02s
            time.sleep(0.02)
            cb.on_step_begin(Mock(), Mock(), Mock())
            # Simulate step taking 0.1s
            cb._step_start_time = time.time() - 0.1
            cb.on_step_end(Mock(), Mock(), Mock())

        # Should have 4 wait times and 5 step times
        assert len(cb._dataloader_wait_times) == 4
        assert len(cb.step_times) == 5

        wait_pct = cb.get_dataloader_wait_pct()
        assert wait_pct is not None
        # Wait time ~0.02s, step time ~0.1s, so wait % should be ~20%
        assert 10.0 < wait_pct < 40.0  # Allow some variance due to timing

    def test_dataloader_wait_tracking_reset_on_train_begin(self):
        """Test that dataloader wait tracking resets on train begin."""
        cb = CanaryCallback()
        cb._dataloader_wait_times = [0.1, 0.2, 0.3]
        cb._step_end_time = 12345.0

        class Mock:
            pass

        cb.on_train_begin(Mock(), Mock(), Mock())

        assert cb._dataloader_wait_times == []
        assert cb._step_end_time is None

    def test_gpu_utilization_reset_on_train_begin(self):
        """Test that GPU utilization tracking resets on train begin."""
        cb = CanaryCallback()
        cb._gpu_utilizations = [80.0, 90.0]

        class Mock:
            pass

        cb.on_train_begin(Mock(), Mock(), Mock())

        assert cb._gpu_utilizations == []

    def test_nvml_init_graceful_failure(self):
        """Test that NVML initialization fails gracefully without pynvml."""
        cb = CanaryCallback()
        # Even without pynvml installed, this should not raise
        # _nvml_available should be False
        # (It might be True if pynvml is installed, which is fine)
        assert isinstance(cb._nvml_available, bool)

    def test_sample_gpu_utilization_no_nvml(self):
        """Test that GPU sampling does nothing when NVML not available."""
        cb = CanaryCallback()
        cb._nvml_available = False

        # Should not raise, should not add data
        initial_len = len(cb._gpu_utilizations)
        cb._sample_gpu_utilization()
        assert len(cb._gpu_utilizations) == initial_len
