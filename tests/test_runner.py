"""Tests for runner configuration and validation."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from canary.runner.base import CanaryConfig, load_config, save_config


class TestCanaryConfig:
    """Tests for CanaryConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CanaryConfig(name="test")

        assert config.name == "test"
        assert config.model_name == "EleutherAI/pythia-70m"
        assert config.training_type == "dpo"
        assert config.max_steps == 200
        assert config.batch_size == 2
        assert config.use_peft is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CanaryConfig(
            name="custom",
            model_name="gpt2",
            training_type="sft",
            max_steps=500,
            batch_size=4,
        )

        assert config.name == "custom"
        assert config.model_name == "gpt2"
        assert config.training_type == "sft"
        assert config.max_steps == 500
        assert config.batch_size == 4

    def test_nested_profiler_config(self):
        """Test nested profiler configuration."""
        config = CanaryConfig(name="test")

        assert config.profiler.enabled is False
        assert config.profiler.start_step == 50
        assert config.profiler.num_steps == 20


class TestConfigIO:
    """Tests for configuration loading and saving."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = CanaryConfig(
            name="test_config",
            model_name="test-model",
            max_steps=100,
        )

        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)

            loaded = load_config(config_path)

            assert loaded.name == "test_config"
            assert loaded.model_name == "test-model"
            assert loaded.max_steps == 100

    def test_load_nonexistent_config(self):
        """Test loading nonexistent configuration raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestRunnerValidation:
    """Tests for runner configuration validation."""

    def test_valid_dpo_config(self):
        """Test validation of valid DPO config."""
        from canary.runner.local import LocalRunner

        config = CanaryConfig(
            name="valid_dpo",
            training_type="dpo",
            max_steps=100,
            batch_size=2,
        )

        runner = LocalRunner(config)
        is_valid, errors = runner.validate_config()

        assert is_valid is True
        assert len(errors) == 0

    def test_valid_ppo_config(self):
        """Test validation of valid PPO config."""
        from canary.runner.local import LocalRunner

        config = CanaryConfig(
            name="valid_ppo",
            training_type="ppo",
            max_steps=50,
            batch_size=2,
            ppo_epochs=4,
            init_kl_coef=0.2,
        )

        runner = LocalRunner(config)
        is_valid, errors = runner.validate_config()

        assert is_valid is True
        assert len(errors) == 0

    def test_ppo_config_defaults(self):
        """Test PPO-specific config defaults."""
        config = CanaryConfig(name="ppo_test", training_type="ppo")

        assert config.ppo_epochs == 4
        assert config.init_kl_coef == 0.2
        assert config.target_kl == 6.0
        assert config.cliprange == 0.2
        assert config.vf_coef == 0.1
        assert config.use_synthetic_reward is True
        assert config.max_new_tokens == 64

    def test_invalid_training_type(self):
        """Test validation rejects invalid training type."""
        from canary.runner.local import LocalRunner

        config = CanaryConfig(
            name="invalid",
            training_type="invalid_type",
        )

        runner = LocalRunner(config)
        is_valid, errors = runner.validate_config()

        assert is_valid is False
        assert any("training_type" in e for e in errors)

    def test_invalid_batch_size(self):
        """Test validation rejects invalid batch size."""
        from canary.runner.local import LocalRunner

        config = CanaryConfig(
            name="invalid",
            batch_size=0,
        )

        runner = LocalRunner(config)
        is_valid, errors = runner.validate_config()

        assert is_valid is False
        assert any("batch_size" in e for e in errors)

    def test_invalid_max_steps(self):
        """Test validation rejects invalid max_steps."""
        from canary.runner.local import LocalRunner

        config = CanaryConfig(
            name="invalid",
            max_steps=0,
        )

        runner = LocalRunner(config)
        is_valid, errors = runner.validate_config()

        assert is_valid is False
        assert any("max_steps" in e for e in errors)
