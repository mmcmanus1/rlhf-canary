"""Tests for comparison and regression detection."""

import pytest

from canary.collect.metrics import (
    CanaryMetrics,
    PerformanceMetrics,
    RunConfig,
    StabilityMetrics,
    StepTimeStats,
)
from canary.compare.stats import CheckStatus, compare_to_baseline
from canary.compare.thresholds import DEFAULT_THRESHOLDS, Thresholds


def make_metrics(
    run_id: str = "test_run",
    step_time_mean: float = 0.5,
    tokens_per_sec: float = 1000.0,
    max_mem_mb: float = 1000.0,
    nan_steps: int = 0,
    inf_steps: int = 0,
    final_loss: float = 0.5,
    loss_diverged: bool = False,
    gpu_name: str = "Test GPU",
) -> CanaryMetrics:
    """Create a test CanaryMetrics object."""
    return CanaryMetrics(
        run_id=run_id,
        timestamp="2024-01-01T00:00:00",
        duration_seconds=100.0,
        env={
            "python_version": "3.10.0",
            "torch_version": "2.0.0",
            "cuda_available": True,
            "cuda_version": "12.0",
            "gpu_name": gpu_name,
            "gpu_count": 1,
            "gpu_memory_gb": 16.0,
            "platform": "Linux",
            "platform_version": "5.0",
            "transformers_version": "4.35.0",
            "trl_version": "0.7.0",
            "fingerprint_hash": "abcd1234",
        },
        config=RunConfig(
            model_name="test-model",
            max_steps=100,
            batch_size=2,
            gradient_accumulation_steps=4,
            max_length=256,
            learning_rate=5e-5,
            beta=0.1,
            dataset_name="test-dataset",
            dataset_size=1000,
            seed=42,
        ),
        perf=PerformanceMetrics(
            step_time=StepTimeStats(
                count=100,
                mean=step_time_mean,
                median=step_time_mean,
                p50=step_time_mean,
                p95=step_time_mean * 1.1,
                p99=step_time_mean * 1.2,
                min=step_time_mean * 0.9,
                max=step_time_mean * 1.3,
                std=0.01,
            ),
            approx_tokens_per_sec=tokens_per_sec,
            max_mem_mb=max_mem_mb,
        ),
        stability=StabilityMetrics(
            nan_steps=nan_steps,
            inf_steps=inf_steps,
            loss_values=[final_loss],
            grad_norm_values=[1.0],
            final_loss=final_loss,
            loss_diverged=loss_diverged,
        ),
        status="completed",
    )


class TestCompareToBaseline:
    """Tests for compare_to_baseline function."""

    def test_identical_metrics_pass(self):
        """Test that identical metrics pass all checks."""
        baseline = make_metrics(run_id="baseline")
        current = make_metrics(run_id="current")

        report = compare_to_baseline(current, baseline)

        assert report.passed is True
        assert len(report.failed_checks) == 0

    def test_step_time_regression_fails(self):
        """Test that step time regression is detected."""
        baseline = make_metrics(run_id="baseline", step_time_mean=0.5)
        current = make_metrics(run_id="current", step_time_mean=0.6)  # 20% increase

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "step_time_mean" in failed_names

    def test_step_time_within_threshold_passes(self):
        """Test that small step time increase passes."""
        baseline = make_metrics(run_id="baseline", step_time_mean=0.5)
        current = make_metrics(run_id="current", step_time_mean=0.52)  # 4% increase

        report = compare_to_baseline(current, baseline)

        step_check = next(c for c in report.checks if c.name == "step_time_mean")
        assert step_check.status == CheckStatus.PASS

    def test_tokens_per_sec_regression_fails(self):
        """Test that tokens/sec regression is detected."""
        baseline = make_metrics(run_id="baseline", tokens_per_sec=1000.0)
        current = make_metrics(run_id="current", tokens_per_sec=850.0)  # 15% drop

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "tokens_per_sec" in failed_names

    def test_memory_regression_fails(self):
        """Test that memory regression is detected."""
        baseline = make_metrics(run_id="baseline", max_mem_mb=1000.0)
        current = make_metrics(run_id="current", max_mem_mb=1600.0)  # +600MB

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "max_memory" in failed_names

    def test_nan_steps_fails(self):
        """Test that NaN steps are detected."""
        baseline = make_metrics(run_id="baseline", nan_steps=0)
        current = make_metrics(run_id="current", nan_steps=5)

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "nan_steps" in failed_names

    def test_inf_steps_fails(self):
        """Test that Inf steps are detected."""
        baseline = make_metrics(run_id="baseline", inf_steps=0)
        current = make_metrics(run_id="current", inf_steps=3)

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "inf_steps" in failed_names

    def test_loss_divergence_fails(self):
        """Test that loss divergence is detected."""
        baseline = make_metrics(run_id="baseline", loss_diverged=False)
        current = make_metrics(run_id="current", loss_diverged=True)

        report = compare_to_baseline(current, baseline)

        assert report.passed is False
        failed_names = [c.name for c in report.failed_checks]
        assert "loss_divergence" in failed_names

    def test_different_gpu_warns(self):
        """Test that different GPU generates warning."""
        baseline = make_metrics(run_id="baseline", gpu_name="GPU A")
        current = make_metrics(run_id="current", gpu_name="GPU B")

        report = compare_to_baseline(current, baseline)

        assert report.env_compatible is False
        assert len(report.env_warnings) > 0

    def test_different_gpu_downgrades_fail_to_warn(self):
        """Test that perf regression with different GPU is a warning, not failure."""
        baseline = make_metrics(run_id="baseline", step_time_mean=0.5, gpu_name="GPU A")
        current = make_metrics(run_id="current", step_time_mean=0.7, gpu_name="GPU B")  # 40% slower

        report = compare_to_baseline(current, baseline)

        # Should not fail because GPUs are different
        step_check = next(c for c in report.checks if c.name == "step_time_mean")
        assert step_check.status == CheckStatus.WARN

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected."""
        baseline = make_metrics(run_id="baseline", step_time_mean=0.5)
        current = make_metrics(run_id="current", step_time_mean=0.6)  # 20% increase

        # Custom threshold allows 25% increase
        thresholds = Thresholds(max_step_time_increase_pct=25.0)

        report = compare_to_baseline(current, baseline, thresholds)

        step_check = next(c for c in report.checks if c.name == "step_time_mean")
        assert step_check.status == CheckStatus.PASS


class TestThresholds:
    """Tests for threshold configurations."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        assert DEFAULT_THRESHOLDS.max_step_time_increase_pct == 10.0
        assert DEFAULT_THRESHOLDS.max_tps_drop_pct == 8.0
        assert DEFAULT_THRESHOLDS.nan_steps_allowed == 0

    def test_threshold_customization(self):
        """Test threshold customization."""
        custom = Thresholds(
            max_step_time_increase_pct=5.0,
            max_tps_drop_pct=3.0,
        )

        assert custom.max_step_time_increase_pct == 5.0
        assert custom.max_tps_drop_pct == 3.0
        # Default values for unset fields
        assert custom.nan_steps_allowed == 0
