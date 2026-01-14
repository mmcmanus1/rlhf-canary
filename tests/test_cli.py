"""Tests for CLI integration."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from canary.cli import main


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_metrics_json():
    """Create sample metrics JSON content."""
    return {
        "run_id": "test_run",
        "timestamp": "2024-01-01T00:00:00",
        "duration_seconds": 100.0,
        "env": {
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
        "config": {
            "model_name": "test-model",
            "max_steps": 100,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_length": 256,
            "learning_rate": 5e-5,
            "beta": 0.1,
            "dataset_name": "test-dataset",
            "dataset_size": 1000,
            "seed": 42,
        },
        "perf": {
            "step_time": {
                "count": 100,
                "mean": 0.5,
                "median": 0.5,
                "p50": 0.5,
                "p95": 0.55,
                "p99": 0.6,
                "min": 0.45,
                "max": 0.65,
                "std": 0.01,
            },
            "approx_tokens_per_sec": 1000.0,
            "max_mem_mb": 1000.0,
            "gpu_utilization_avg": 85.0,
            "dataloader_wait_pct": 5.0,
        },
        "stability": {
            "nan_steps": 0,
            "inf_steps": 0,
            "loss_values": [0.5],
            "grad_norm_values": [1.0],
            "final_loss": 0.5,
            "loss_diverged": False,
        },
        "status": "completed",
    }


class TestMainGroup:
    """Tests for the main CLI group."""

    def test_help(self, runner):
        """Test --help option."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "RLHF Canary" in result.output

    def test_version(self, runner):
        """Test --version option."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()


class TestRunCommand:
    """Tests for the 'run' command."""

    def test_run_help(self, runner):
        """Test run --help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output

    def test_run_missing_config(self, runner):
        """Test run with missing config file."""
        result = runner.invoke(main, ["run", "nonexistent.yaml"])
        assert result.exit_code != 0
        assert "does not exist" in result.output.lower() or "no such file" in result.output.lower()


class TestCompareCommand:
    """Tests for the 'compare' command."""

    def test_compare_help(self, runner):
        """Test compare --help."""
        result = runner.invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "CURRENT_PATH" in result.output
        assert "BASELINE_PATH" in result.output

    def test_compare_missing_files(self, runner):
        """Test compare with missing files."""
        result = runner.invoke(main, ["compare", "current.json", "baseline.json"])
        assert result.exit_code != 0

    def test_compare_identical_metrics(self, runner, sample_metrics_json):
        """Test compare with identical metrics (should pass)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write identical metrics to both files
            current_path = Path(tmpdir) / "current.json"
            baseline_path = Path(tmpdir) / "baseline.json"

            current_path.write_text(json.dumps(sample_metrics_json))
            baseline_path.write_text(json.dumps(sample_metrics_json))

            result = runner.invoke(
                main, ["compare", str(current_path), str(baseline_path)]
            )
            assert result.exit_code == 0
            assert "All checks passed" in result.output

    def test_compare_regression_detected(self, runner, sample_metrics_json):
        """Test compare with regression (should fail)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create baseline
            baseline_path = Path(tmpdir) / "baseline.json"
            baseline_path.write_text(json.dumps(sample_metrics_json))

            # Create current with regression (slower step time)
            current_metrics = sample_metrics_json.copy()
            current_metrics["perf"] = sample_metrics_json["perf"].copy()
            current_metrics["perf"]["step_time"] = sample_metrics_json["perf"][
                "step_time"
            ].copy()
            current_metrics["perf"]["step_time"]["mean"] = 0.7  # 40% slower

            current_path = Path(tmpdir) / "current.json"
            current_path.write_text(json.dumps(current_metrics))

            result = runner.invoke(
                main, ["compare", str(current_path), str(baseline_path)]
            )
            assert result.exit_code == 1
            assert "Regression detected" in result.output

    def test_compare_json_output(self, runner, sample_metrics_json):
        """Test compare with --json output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "current.json"
            baseline_path = Path(tmpdir) / "baseline.json"

            current_path.write_text(json.dumps(sample_metrics_json))
            baseline_path.write_text(json.dumps(sample_metrics_json))

            result = runner.invoke(
                main, ["compare", str(current_path), str(baseline_path), "--json"]
            )
            assert result.exit_code == 0
            # Output contains JSON embedded in CLI messages
            # Find the JSON part by looking for first { and last }
            output = result.output
            json_start = output.find("{")
            json_end = output.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                output_json = json.loads(output[json_start:json_end])
                assert "passed" in output_json

    def test_compare_with_output_file(self, runner, sample_metrics_json):
        """Test compare with -o output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "current.json"
            baseline_path = Path(tmpdir) / "baseline.json"
            output_path = Path(tmpdir) / "report.md"

            current_path.write_text(json.dumps(sample_metrics_json))
            baseline_path.write_text(json.dumps(sample_metrics_json))

            result = runner.invoke(
                main,
                [
                    "compare",
                    str(current_path),
                    str(baseline_path),
                    "-o",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0
            assert output_path.exists()
            assert "RLHF Canary Report" in output_path.read_text()


class TestSaveBaselineCommand:
    """Tests for the 'save-baseline' command."""

    def test_save_baseline_help(self, runner):
        """Test save-baseline --help."""
        result = runner.invoke(main, ["save-baseline", "--help"])
        assert result.exit_code == 0
        assert "METRICS_PATH" in result.output

    def test_save_baseline(self, runner, sample_metrics_json):
        """Test saving a baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.json"
            baseline_path = Path(tmpdir) / "baselines" / "baseline.json"

            metrics_path.write_text(json.dumps(sample_metrics_json))

            result = runner.invoke(
                main, ["save-baseline", str(metrics_path), str(baseline_path)]
            )
            assert result.exit_code == 0
            assert baseline_path.exists()
            assert "Saved baseline" in result.output


class TestEnvCommand:
    """Tests for the 'env' command."""

    def test_env(self, runner):
        """Test env command shows fingerprint."""
        result = runner.invoke(main, ["env"])
        assert result.exit_code == 0
        assert "Environment Fingerprint" in result.output
        assert "Python" in result.output
        assert "PyTorch" in result.output


class TestInitConfigCommand:
    """Tests for the 'init-config' command."""

    def test_init_config_help(self, runner):
        """Test init-config --help."""
        result = runner.invoke(main, ["init-config", "--help"])
        assert result.exit_code == 0
        assert "OUTPUT_PATH" in result.output

    def test_init_config_dpo_smoke(self, runner):
        """Test creating DPO smoke config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            result = runner.invoke(
                main, ["init-config", str(output_path), "--type", "dpo", "--tier", "smoke"]
            )
            assert result.exit_code == 0
            assert output_path.exists()
            assert "Created config" in result.output

            # Verify content
            import yaml

            config = yaml.safe_load(output_path.read_text())
            assert config["training_type"] == "dpo"
            assert config["max_steps"] == 100
            assert "beta" in config

    def test_init_config_sft_nightly(self, runner):
        """Test creating SFT nightly config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            result = runner.invoke(
                main,
                ["init-config", str(output_path), "--type", "sft", "--tier", "nightly"],
            )
            assert result.exit_code == 0
            assert output_path.exists()

            import yaml

            config = yaml.safe_load(output_path.read_text())
            assert config["training_type"] == "sft"
            assert config["max_steps"] == 2000
            assert "beta" not in config


class TestGhReportCommand:
    """Tests for the 'gh-report' command."""

    def test_gh_report_help(self, runner):
        """Test gh-report --help."""
        result = runner.invoke(main, ["gh-report", "--help"])
        assert result.exit_code == 0
        assert "CURRENT_PATH" in result.output
        assert "BASELINE_PATH" in result.output

    def test_gh_report_with_output(self, runner, sample_metrics_json):
        """Test gh-report with output file (no actual GitHub posting)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            current_path = Path(tmpdir) / "current.json"
            baseline_path = Path(tmpdir) / "baseline.json"
            output_path = Path(tmpdir) / "report.md"

            current_path.write_text(json.dumps(sample_metrics_json))
            baseline_path.write_text(json.dumps(sample_metrics_json))

            result = runner.invoke(
                main,
                [
                    "gh-report",
                    str(current_path),
                    str(baseline_path),
                    "--no-comment",
                    "--no-status",
                    "-o",
                    str(output_path),
                ],
            )
            assert result.exit_code == 0
            assert output_path.exists()


class TestVerboseFlag:
    """Tests for the -v/--verbose flag."""

    def test_verbose_flag(self, runner):
        """Test that -v flag is accepted."""
        result = runner.invoke(main, ["-v", "env"])
        assert result.exit_code == 0
