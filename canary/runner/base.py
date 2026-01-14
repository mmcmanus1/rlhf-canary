"""Base runner interface for canary jobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from canary.collect.metrics import CanaryMetrics
from canary.collect.profiler import ProfilerConfig


class CanaryConfig(BaseModel):
    """Configuration for a canary training run."""

    # Run identification
    name: str
    description: str = ""

    # Model configuration
    model_name: str = "EleutherAI/pythia-70m"
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Quantization (optional)
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Training configuration
    training_type: str = "dpo"  # sft, dpo, ppo
    max_steps: int = 200
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    max_length: int = 256
    warmup_steps: int = 10

    # DPO-specific
    beta: float = 0.1
    max_prompt_length: int = 64

    # Dataset configuration
    dataset_name: str = "Anthropic/hh-rlhf"
    dataset_split: str = "train"
    dataset_size: int = 1024
    seed: int = 42

    # Output configuration
    output_dir: str = "./canary_output"

    # Profiler configuration
    profiler: ProfilerConfig = ProfilerConfig()

    # Metrics configuration
    metrics_warmup_steps: int = 10

    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class RunResult(BaseModel):
    """Result from a canary run."""

    success: bool
    metrics: CanaryMetrics | None = None
    error: str | None = None
    metrics_path: str | None = None
    logs_path: str | None = None


class BaseRunner(ABC):
    """Abstract base class for canary job runners."""

    def __init__(self, config: CanaryConfig):
        """Initialize runner with configuration.

        Args:
            config: Canary run configuration.
        """
        self.config = config

    @abstractmethod
    def run(self) -> RunResult:
        """Execute the canary training run.

        Returns:
            RunResult with metrics or error information.
        """
        pass

    @abstractmethod
    def validate_config(self) -> tuple[bool, list[str]]:
        """Validate the configuration before running.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        pass

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import hashlib
        import time

        timestamp = int(time.time())
        config_hash = hashlib.sha256(self.config.model_dump_json().encode()).hexdigest()[:8]
        return f"{self.config.name}_{timestamp}_{config_hash}"


def load_config(config_path: str | Path) -> CanaryConfig:
    """Load canary configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        CanaryConfig loaded from the file.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return CanaryConfig(**config_dict)


def save_config(config: CanaryConfig, output_path: str | Path) -> None:
    """Save canary configuration to a YAML file.

    Args:
        config: Configuration to save.
        output_path: Path to save the configuration.
    """
    import yaml

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)
