"""Tests for report generation."""

import pytest

from canary.collect.metrics import (
    CanaryMetrics,
    PerformanceMetrics,
    RunConfig,
    StabilityMetrics,
    StepTimeStats,
)
from canary.compare.stats import Check, CheckStatus, ComparisonReport
from canary.report.markdown import (
    _format_change,
    _format_value,
    _get_status_icon,
    generate_markdown_report,
    generate_short_summary,
)


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
    beta: float | None = 0.1,
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
            beta=beta,
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


def make_passing_report() -> ComparisonReport:
    """Create a passing comparison report."""
    return ComparisonReport(
        passed=True,
        baseline_run_id="baseline_run",
        current_run_id="current_run",
        env_compatible=True,
        env_warnings=[],
        warnings=[],
        checks=[
            Check(
                name="step_time_mean",
                status=CheckStatus.PASS,
                baseline_value=0.5,
                current_value=0.52,
                delta=0.02,
                delta_pct=4.0,
                threshold=10.0,
                message="Step time within threshold",
            ),
            Check(
                name="tokens_per_sec",
                status=CheckStatus.PASS,
                baseline_value=1000.0,
                current_value=980.0,
                delta=-20.0,
                delta_pct=-2.0,
                threshold=8.0,
                message="Throughput within threshold",
            ),
            Check(
                name="max_memory",
                status=CheckStatus.PASS,
                baseline_value=1000.0,
                current_value=1050.0,
                delta=50.0,
                delta_pct=5.0,
                threshold=500.0,
                message="Memory within threshold",
            ),
        ],
    )


def make_failing_report() -> ComparisonReport:
    """Create a failing comparison report."""
    return ComparisonReport(
        passed=False,
        baseline_run_id="baseline_run",
        current_run_id="current_run",
        env_compatible=True,
        env_warnings=[],
        warnings=[],
        checks=[
            Check(
                name="step_time_mean",
                status=CheckStatus.FAIL,
                baseline_value=0.5,
                current_value=0.7,
                delta=0.2,
                delta_pct=40.0,
                threshold=10.0,
                message="Step time increased by 40.0% (threshold: 10.0%)",
            ),
            Check(
                name="tokens_per_sec",
                status=CheckStatus.FAIL,
                baseline_value=1000.0,
                current_value=700.0,
                delta=-300.0,
                delta_pct=-30.0,
                threshold=8.0,
                message="Throughput dropped by 30.0% (threshold: 8.0%)",
            ),
        ],
    )


class TestFormatValue:
    """Tests for _format_value helper."""

    def test_none_value(self):
        """Test None returns dash."""
        assert _format_value(None) == "-"

    def test_large_value(self):
        """Test large values are formatted with commas."""
        result = _format_value(1500.0)
        assert result == "1,500"

    def test_medium_value(self):
        """Test medium values have 2 decimal places."""
        assert _format_value(12.345) == "12.35"

    def test_small_value(self):
        """Test small values have 4 decimal places."""
        assert _format_value(0.0123) == "0.0123"


class TestFormatChange:
    """Tests for _format_change helper."""

    def test_both_none(self):
        """Test both None returns dash."""
        assert _format_change(None, None) == "-"

    def test_percentage_preferred(self):
        """Test percentage is preferred over absolute."""
        result = _format_change(10.0, 25.5)
        assert result == "+25.5%"

    def test_negative_percentage(self):
        """Test negative percentage formatting."""
        result = _format_change(-10.0, -15.3)
        assert result == "-15.3%"

    def test_absolute_only(self):
        """Test absolute value when no percentage."""
        result = _format_change(5.5, None)
        assert result == "+5.50"


class TestGetStatusIcon:
    """Tests for _get_status_icon helper."""

    def test_pass_icon(self):
        """Test PASS status icon."""
        assert _get_status_icon(CheckStatus.PASS) == "✅"

    def test_fail_icon(self):
        """Test FAIL status icon."""
        assert _get_status_icon(CheckStatus.FAIL) == "❌"

    def test_warn_icon(self):
        """Test WARN status icon."""
        assert _get_status_icon(CheckStatus.WARN) == "⚠️"

    def test_skip_icon(self):
        """Test SKIP status icon."""
        assert _get_status_icon(CheckStatus.SKIP) == "⏭️"


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    def test_passing_report_header(self):
        """Test passing report has correct header."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "# RLHF Canary Report ✅" in markdown
        assert "**Status:** PASS" in markdown

    def test_failing_report_header(self):
        """Test failing report has correct header."""
        report = make_failing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run", step_time_mean=0.7)

        markdown = generate_markdown_report(report, current, baseline)

        assert "# RLHF Canary Report ❌" in markdown
        assert "**Status:** FAIL" in markdown

    def test_report_includes_run_ids(self):
        """Test report includes run IDs."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "baseline_run" in markdown
        assert "current_run" in markdown

    def test_report_includes_checks_table(self):
        """Test report includes checks table."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "## Regression Checks" in markdown
        assert "| Check | Status | Baseline | Current | Change | Threshold |" in markdown
        assert "step_time_mean" in markdown

    def test_failing_report_includes_failed_checks_section(self):
        """Test failing report has Failed Checks section."""
        report = make_failing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run", step_time_mean=0.7)

        markdown = generate_markdown_report(report, current, baseline)

        assert "## Failed Checks" in markdown
        assert "❌ step_time_mean" in markdown

    def test_report_includes_performance_summary(self):
        """Test report includes performance summary."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "## Performance Summary" in markdown
        assert "Step Time (mean)" in markdown
        assert "Tokens/sec" in markdown
        assert "Peak Memory" in markdown

    def test_report_includes_configuration(self):
        """Test report includes configuration section."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "## Configuration" in markdown
        assert "test-model" in markdown
        assert "Batch Size" in markdown

    def test_report_includes_beta_for_dpo(self):
        """Test report includes beta for DPO configs."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run", beta=0.1)
        current = make_metrics(run_id="current_run", beta=0.1)

        markdown = generate_markdown_report(report, current, baseline)

        assert "Beta (DPO)" in markdown

    def test_report_excludes_beta_for_sft(self):
        """Test report excludes beta for SFT configs."""
        report = make_passing_report()
        baseline = make_metrics(run_id="baseline_run", beta=None)
        current = make_metrics(run_id="current_run", beta=None)

        markdown = generate_markdown_report(report, current, baseline)

        assert "Beta (DPO)" not in markdown

    def test_env_warning_shown(self):
        """Test environment warning is shown when applicable."""
        report = ComparisonReport(
            passed=True,
            baseline_run_id="baseline_run",
            current_run_id="current_run",
            env_compatible=False,
            env_warnings=["GPU changed: GPU A -> GPU B"],
            warnings=[],
            checks=[],
        )
        baseline = make_metrics(run_id="baseline_run", gpu_name="GPU A")
        current = make_metrics(run_id="current_run", gpu_name="GPU B")

        markdown = generate_markdown_report(report, current, baseline)

        assert "Environment Warning" in markdown
        assert "GPU changed" in markdown

    def test_heuristics_included_on_failure(self):
        """Test root cause analysis is included on failure."""
        report = make_failing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run", step_time_mean=0.7)

        markdown = generate_markdown_report(report, current, baseline, include_heuristics=True)

        assert "Root Cause Analysis" in markdown

    def test_heuristics_excluded_when_disabled(self):
        """Test root cause analysis is excluded when disabled."""
        report = make_failing_report()
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run", step_time_mean=0.7)

        markdown = generate_markdown_report(report, current, baseline, include_heuristics=False)

        assert "Root Cause Analysis" not in markdown


class TestGenerateShortSummary:
    """Tests for generate_short_summary function."""

    def test_passing_summary(self):
        """Test passing summary format."""
        report = make_passing_report()
        summary = generate_short_summary(report)

        assert "PASS ✅" in summary

    def test_failing_summary_lists_failures(self):
        """Test failing summary lists failed checks."""
        report = make_failing_report()
        summary = generate_short_summary(report)

        assert "FAIL ❌" in summary
        assert "step_time_mean" in summary
        assert "tokens_per_sec" in summary


class TestWarningChecks:
    """Tests for warning check handling in reports."""

    def test_warning_checks_shown(self):
        """Test warning checks are shown in report."""
        report = ComparisonReport(
            passed=True,
            baseline_run_id="baseline_run",
            current_run_id="current_run",
            env_compatible=False,
            env_warnings=["Different GPU"],
            warnings=[],
            checks=[
                Check(
                    name="step_time_mean",
                    status=CheckStatus.WARN,
                    baseline_value=0.5,
                    current_value=0.6,
                    delta=0.1,
                    delta_pct=20.0,
                    threshold=10.0,
                    message="Step time increased but GPU differs",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline_run")
        current = make_metrics(run_id="current_run")

        markdown = generate_markdown_report(report, current, baseline)

        assert "## Warnings" in markdown
        assert "step_time_mean" in markdown
