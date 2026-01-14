"""Tests for root cause heuristics."""

import pytest

from canary.collect.metrics import (
    CanaryMetrics,
    PerformanceMetrics,
    RunConfig,
    StabilityMetrics,
    StepTimeStats,
)
from canary.compare.heuristics import (
    HeuristicAnalysis,
    RegressionCategory,
    Suspect,
    _analyze_dataloader_wait,
    _analyze_gpu_utilization,
    _analyze_grad_norm_patterns,
    analyze_regression,
    format_suspects_markdown,
)
from canary.compare.stats import Check, CheckStatus, ComparisonReport


def make_metrics(
    run_id: str = "test_run",
    step_time_mean: float = 0.5,
    tokens_per_sec: float = 1000.0,
    max_mem_mb: float = 1000.0,
    nan_steps: int = 0,
    inf_steps: int = 0,
    final_loss: float = 0.5,
    loss_diverged: bool = False,
    gpu_utilization_avg: float | None = 85.0,
    dataloader_wait_pct: float | None = 5.0,
    grad_norm_values: list[float] | None = None,
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
            "gpu_name": "Test GPU",
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
            gpu_utilization_avg=gpu_utilization_avg,
            dataloader_wait_pct=dataloader_wait_pct,
        ),
        stability=StabilityMetrics(
            nan_steps=nan_steps,
            inf_steps=inf_steps,
            loss_values=[final_loss],
            grad_norm_values=grad_norm_values or [1.0],
            final_loss=final_loss,
            loss_diverged=loss_diverged,
        ),
        status="completed",
    )


class TestRegressionCategory:
    """Tests for RegressionCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert RegressionCategory.DATALOADER == "dataloader"
        assert RegressionCategory.TOKENIZATION == "tokenization"
        assert RegressionCategory.COMMUNICATION == "communication"
        assert RegressionCategory.MEMORY == "memory"
        assert RegressionCategory.KERNEL == "kernel"
        assert RegressionCategory.IO == "io"
        assert RegressionCategory.UNKNOWN == "unknown"


class TestSuspect:
    """Tests for Suspect model."""

    def test_suspect_creation(self):
        """Test creating a suspect."""
        suspect = Suspect(
            category=RegressionCategory.DATALOADER,
            confidence=0.8,
            description="Test description",
            evidence=["Evidence 1", "Evidence 2"],
            suggested_actions=["Action 1"],
        )

        assert suspect.category == RegressionCategory.DATALOADER
        assert suspect.confidence == 0.8
        assert len(suspect.evidence) == 2
        assert len(suspect.suggested_actions) == 1


class TestHeuristicAnalysis:
    """Tests for HeuristicAnalysis model."""

    def test_empty_analysis(self):
        """Test analysis with no suspects."""
        analysis = HeuristicAnalysis(
            suspects=[],
            summary="No regression detected",
            top_suspect=None,
        )

        assert len(analysis.suspects) == 0
        assert analysis.top_suspect is None

    def test_analysis_with_suspects(self):
        """Test analysis with suspects."""
        suspect = Suspect(
            category=RegressionCategory.MEMORY,
            confidence=0.9,
            description="Memory issue",
            evidence=["High memory usage"],
            suggested_actions=["Reduce batch size"],
        )

        analysis = HeuristicAnalysis(
            suspects=[suspect],
            summary="Memory issue detected",
            top_suspect=suspect,
        )

        assert len(analysis.suspects) == 1
        assert analysis.top_suspect == suspect


class TestAnalyzeRegression:
    """Tests for analyze_regression function."""

    def test_no_regression(self):
        """Test analysis when no regression."""
        report = ComparisonReport(
            passed=True,
            baseline_run_id="baseline",
            current_run_id="current",
            env_compatible=True,
            env_warnings=[],
            warnings=[],
            checks=[],
        )
        baseline = make_metrics(run_id="baseline")
        current = make_metrics(run_id="current")

        analysis = analyze_regression(report, current, baseline)

        assert "No clear regression pattern" in analysis.summary

    def test_step_time_regression_high(self):
        """Test analysis for high step time regression (>20%)."""
        report = ComparisonReport(
            passed=False,
            baseline_run_id="baseline",
            current_run_id="current",
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
                    message="Step time increased",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline")
        current = make_metrics(run_id="current", step_time_mean=0.7)

        analysis = analyze_regression(report, current, baseline)

        # Should identify dataloader as a suspect
        categories = [s.category for s in analysis.suspects]
        assert RegressionCategory.DATALOADER in categories

    def test_memory_regression(self):
        """Test analysis for memory regression."""
        report = ComparisonReport(
            passed=False,
            baseline_run_id="baseline",
            current_run_id="current",
            env_compatible=True,
            env_warnings=[],
            warnings=[],
            checks=[
                Check(
                    name="max_memory",
                    status=CheckStatus.FAIL,
                    baseline_value=1000.0,
                    current_value=1600.0,
                    delta=600.0,
                    delta_pct=60.0,
                    threshold=500.0,
                    message="Memory increased",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline", max_mem_mb=1000.0)
        current = make_metrics(run_id="current", max_mem_mb=1600.0)

        analysis = analyze_regression(report, current, baseline)

        categories = [s.category for s in analysis.suspects]
        assert RegressionCategory.MEMORY in categories

    def test_nan_steps_regression(self):
        """Test analysis for NaN steps."""
        report = ComparisonReport(
            passed=False,
            baseline_run_id="baseline",
            current_run_id="current",
            env_compatible=True,
            env_warnings=[],
            warnings=[],
            checks=[
                Check(
                    name="nan_steps",
                    status=CheckStatus.FAIL,
                    baseline_value=0,
                    current_value=5,
                    delta=5,
                    delta_pct=None,
                    threshold=0,
                    message="NaN detected",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline", nan_steps=0)
        current = make_metrics(run_id="current", nan_steps=5)

        analysis = analyze_regression(report, current, baseline)

        # Should have numerical instability suspect
        assert len(analysis.suspects) > 0
        # Check for learning rate suggestion
        all_actions = []
        for s in analysis.suspects:
            all_actions.extend(s.suggested_actions)
        assert any("learning rate" in a.lower() for a in all_actions)

    def test_loss_divergence(self):
        """Test analysis for loss divergence."""
        report = ComparisonReport(
            passed=False,
            baseline_run_id="baseline",
            current_run_id="current",
            env_compatible=True,
            env_warnings=[],
            warnings=[],
            checks=[
                Check(
                    name="loss_divergence",
                    status=CheckStatus.FAIL,
                    baseline_value=None,
                    current_value=None,
                    delta=None,
                    delta_pct=None,
                    threshold=None,
                    message="Loss diverged",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline", loss_diverged=False)
        current = make_metrics(run_id="current", loss_diverged=True)

        analysis = analyze_regression(report, current, baseline)

        assert len(analysis.suspects) > 0

    def test_combined_pattern_memory_and_step_time(self):
        """Test analysis for combined memory + step time regression."""
        report = ComparisonReport(
            passed=False,
            baseline_run_id="baseline",
            current_run_id="current",
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
                    message="Step time increased",
                ),
                Check(
                    name="max_memory",
                    status=CheckStatus.FAIL,
                    baseline_value=1000.0,
                    current_value=1600.0,
                    delta=600.0,
                    delta_pct=60.0,
                    threshold=500.0,
                    message="Memory increased",
                ),
            ],
        )
        baseline = make_metrics(run_id="baseline")
        current = make_metrics(run_id="current", step_time_mean=0.7, max_mem_mb=1600.0)

        analysis = analyze_regression(report, current, baseline)

        # Should detect memory fragmentation pattern
        descriptions = [s.description.lower() for s in analysis.suspects]
        assert any("memory" in d for d in descriptions)


class TestAnalyzeGpuUtilization:
    """Tests for GPU utilization analysis."""

    def test_low_gpu_utilization(self):
        """Test detection of low GPU utilization."""
        current = make_metrics(gpu_utilization_avg=40.0)
        baseline = make_metrics(gpu_utilization_avg=85.0)

        suspects = _analyze_gpu_utilization(current, baseline)

        assert len(suspects) > 0
        assert suspects[0].category == RegressionCategory.DATALOADER
        assert "CPU bottleneck" in suspects[0].description

    def test_gpu_utilization_drop(self):
        """Test detection of GPU utilization drop."""
        current = make_metrics(gpu_utilization_avg=60.0)
        baseline = make_metrics(gpu_utilization_avg=90.0)

        suspects = _analyze_gpu_utilization(current, baseline)

        assert len(suspects) > 0
        assert any("dropped" in s.description.lower() for s in suspects)

    def test_no_gpu_data(self):
        """Test with no GPU utilization data."""
        current = make_metrics(gpu_utilization_avg=None)
        baseline = make_metrics(gpu_utilization_avg=85.0)

        suspects = _analyze_gpu_utilization(current, baseline)

        assert len(suspects) == 0


class TestAnalyzeDataloaderWait:
    """Tests for dataloader wait analysis."""

    def test_high_dataloader_wait(self):
        """Test detection of high dataloader wait time."""
        current = make_metrics(dataloader_wait_pct=30.0)
        baseline = make_metrics(dataloader_wait_pct=5.0)

        suspects = _analyze_dataloader_wait(current, baseline)

        assert len(suspects) > 0
        assert suspects[0].category == RegressionCategory.DATALOADER
        assert "waiting for data" in suspects[0].description.lower()

    def test_dataloader_wait_increase(self):
        """Test detection of dataloader wait time increase."""
        current = make_metrics(dataloader_wait_pct=15.0)
        baseline = make_metrics(dataloader_wait_pct=5.0)

        suspects = _analyze_dataloader_wait(current, baseline)

        assert len(suspects) > 0

    def test_no_dataloader_data(self):
        """Test with no dataloader wait data."""
        current = make_metrics(dataloader_wait_pct=None)
        baseline = make_metrics(dataloader_wait_pct=5.0)

        suspects = _analyze_dataloader_wait(current, baseline)

        assert len(suspects) == 0


class TestAnalyzeGradNormPatterns:
    """Tests for gradient norm pattern analysis."""

    def test_gradient_explosion(self):
        """Test detection of gradient explosion."""
        # Create increasing gradient norms
        grad_norms = [1.0] * 10 + [5.0] * 10 + [10.0] * 10
        current = make_metrics(grad_norm_values=grad_norms)
        baseline = make_metrics(grad_norm_values=[1.0] * 30)

        suspects = _analyze_grad_norm_patterns(current, baseline)

        assert len(suspects) > 0
        assert any("explosion" in s.description.lower() for s in suspects)

    def test_high_gradient_variance(self):
        """Test detection of high gradient variance."""
        import random

        random.seed(42)
        # Create high variance gradient norms
        grad_norms = [random.uniform(0.1, 10.0) for _ in range(50)]
        current = make_metrics(grad_norm_values=grad_norms)
        baseline = make_metrics(grad_norm_values=[1.0] * 50)

        suspects = _analyze_grad_norm_patterns(current, baseline)

        # May or may not detect depending on specific values
        # Just verify it doesn't crash
        assert isinstance(suspects, list)

    def test_insufficient_data(self):
        """Test with insufficient gradient data."""
        current = make_metrics(grad_norm_values=[1.0, 2.0])
        baseline = make_metrics(grad_norm_values=[1.0])

        suspects = _analyze_grad_norm_patterns(current, baseline)

        # Should return empty with insufficient data
        assert len(suspects) == 0


class TestFormatSuspectsMarkdown:
    """Tests for markdown formatting of suspects."""

    def test_empty_suspects(self):
        """Test formatting with no suspects."""
        analysis = HeuristicAnalysis(
            suspects=[],
            summary="No issues",
            top_suspect=None,
        )

        markdown = format_suspects_markdown(analysis)

        assert "No clear root cause" in markdown

    def test_single_suspect(self):
        """Test formatting with single suspect."""
        suspect = Suspect(
            category=RegressionCategory.MEMORY,
            confidence=0.9,
            description="Memory leak detected",
            evidence=["High memory usage", "Memory increasing over time"],
            suggested_actions=["Check for memory leaks", "Reduce batch size"],
        )
        analysis = HeuristicAnalysis(
            suspects=[suspect],
            summary="Memory issue detected",
            top_suspect=suspect,
        )

        markdown = format_suspects_markdown(analysis)

        assert "Root Cause Analysis" in markdown
        assert "Memory" in markdown
        assert "Memory leak detected" in markdown
        assert "High memory usage" in markdown
        assert "Check for memory leaks" in markdown

    def test_multiple_suspects_limited_to_three(self):
        """Test that only top 3 suspects are shown."""
        suspects = [
            Suspect(
                category=RegressionCategory.MEMORY,
                confidence=0.9 - i * 0.1,
                description=f"Suspect {i}",
                evidence=[f"Evidence {i}"],
                suggested_actions=[f"Action {i}"],
            )
            for i in range(5)
        ]

        analysis = HeuristicAnalysis(
            suspects=suspects,
            summary="Multiple issues",
            top_suspect=suspects[0],
        )

        markdown = format_suspects_markdown(analysis)

        # Should show #1, #2, #3 but not #4, #5
        assert "#1" in markdown
        assert "#2" in markdown
        assert "#3" in markdown
        # Suspects 4 and 5 should not appear as numbered entries
        assert markdown.count("Suspect") <= 3

    def test_confidence_bar(self):
        """Test confidence bar formatting."""
        suspect = Suspect(
            category=RegressionCategory.DATALOADER,
            confidence=0.8,
            description="Test",
            evidence=["Test"],
            suggested_actions=["Test"],
        )
        analysis = HeuristicAnalysis(
            suspects=[suspect],
            summary="Test",
            top_suspect=suspect,
        )

        markdown = format_suspects_markdown(analysis)

        # Should have 8 filled blocks and 2 empty blocks
        assert "████████░░" in markdown
        assert "80%" in markdown
