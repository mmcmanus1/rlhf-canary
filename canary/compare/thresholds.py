"""Threshold definitions for regression detection."""

from __future__ import annotations

from pydantic import BaseModel


class Thresholds(BaseModel):
    """Configurable thresholds for regression detection."""

    # Performance thresholds
    max_step_time_increase_pct: float = 10.0  # Fail if step time increases > 10%
    max_tps_drop_pct: float = 8.0  # Fail if tokens/sec drops > 8%
    max_mem_increase_mb: float = 500.0  # Fail if memory increases > 500MB
    max_mem_increase_pct: float = 20.0  # Fail if memory increases > 20%

    # Stability thresholds
    nan_steps_allowed: int = 0  # Fail if any NaN steps
    inf_steps_allowed: int = 0  # Fail if any Inf steps
    max_loss_increase_pct: float = 50.0  # Warn if final loss > 50% higher

    # Statistical significance
    min_step_count: int = 20  # Minimum steps required for valid comparison
    confidence_level: float = 0.95  # Confidence level for statistical tests


# Default thresholds for different test tiers
DEFAULT_THRESHOLDS = Thresholds()

SMOKE_THRESHOLDS = Thresholds(
    max_step_time_increase_pct=15.0,  # More lenient for short runs
    max_tps_drop_pct=12.0,
    max_mem_increase_mb=1000.0,
    min_step_count=10,
)

PERF_THRESHOLDS = Thresholds(
    max_step_time_increase_pct=8.0,  # Stricter for perf tests
    max_tps_drop_pct=5.0,
    max_mem_increase_mb=300.0,
    min_step_count=50,
)

NIGHTLY_THRESHOLDS = Thresholds(
    max_step_time_increase_pct=5.0,  # Strictest for nightly
    max_tps_drop_pct=3.0,
    max_mem_increase_mb=200.0,
    min_step_count=100,
)


def get_thresholds(tier: str = "default") -> Thresholds:
    """Get thresholds for a given test tier.

    Args:
        tier: One of "default", "smoke", "perf", "nightly".

    Returns:
        Thresholds for the specified tier.
    """
    thresholds_map = {
        "default": DEFAULT_THRESHOLDS,
        "smoke": SMOKE_THRESHOLDS,
        "perf": PERF_THRESHOLDS,
        "nightly": NIGHTLY_THRESHOLDS,
    }
    return thresholds_map.get(tier, DEFAULT_THRESHOLDS)
