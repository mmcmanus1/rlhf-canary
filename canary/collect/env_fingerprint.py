"""Environment fingerprinting for reproducible comparisons."""

from __future__ import annotations

import hashlib
import platform
from typing import Any

from pydantic import BaseModel


class EnvFingerprint(BaseModel):
    """Environment fingerprint for baseline comparison."""

    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_count: int = 0
    gpu_memory_gb: float | None = None
    platform: str
    platform_version: str
    transformers_version: str | None = None
    trl_version: str | None = None
    fingerprint_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute fingerprint hash after initialization."""
        if not self.fingerprint_hash:
            # Create hash from hardware-relevant fields only
            key_fields = f"{self.gpu_name}:{self.cuda_version}:{self.gpu_count}"
            self.fingerprint_hash = hashlib.md5(key_fields.encode()).hexdigest()[:8]


def _safe_import_version(module_name: str) -> str | None:
    """Safely get version of an optional module."""
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except ImportError:
        return None


def get_env_fingerprint() -> EnvFingerprint:
    """Collect environment fingerprint for baseline comparison.

    Returns:
        EnvFingerprint with system, CUDA, and library version info.
    """
    # Torch info
    torch_version = "N/A"
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_count = 0
    gpu_memory_gb = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            cuda_version = torch.version.cuda
            gpu_count = torch.cuda.device_count()

            if gpu_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = props.total_memory / (1024**3)
    except ImportError:
        pass

    return EnvFingerprint(
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
        platform=platform.system(),
        platform_version=platform.release(),
        transformers_version=_safe_import_version("transformers"),
        trl_version=_safe_import_version("trl"),
    )


def fingerprints_compatible(a: EnvFingerprint, b: EnvFingerprint) -> tuple[bool, list[str]]:
    """Check if two fingerprints are compatible for comparison.

    Returns:
        Tuple of (is_compatible, list of warnings).
    """
    warnings = []

    # GPU must match for meaningful perf comparison
    if a.gpu_name != b.gpu_name:
        warnings.append(
            f"GPU mismatch: baseline={a.gpu_name}, current={b.gpu_name}. "
            "Performance comparisons may be unreliable."
        )

    # GPU count difference
    if a.gpu_count != b.gpu_count:
        warnings.append(
            f"GPU count mismatch: baseline={a.gpu_count}, current={b.gpu_count}."
        )

    # CUDA version difference (warn but don't fail)
    if a.cuda_version != b.cuda_version:
        warnings.append(
            f"CUDA version mismatch: baseline={a.cuda_version}, current={b.cuda_version}."
        )

    # PyTorch major version difference
    if a.torch_version.split(".")[0] != b.torch_version.split(".")[0]:
        warnings.append(
            f"PyTorch major version mismatch: baseline={a.torch_version}, "
            f"current={b.torch_version}."
        )

    # Compatible if GPU matches (most important factor for perf)
    is_compatible = a.gpu_name == b.gpu_name and a.gpu_count == b.gpu_count

    return is_compatible, warnings
