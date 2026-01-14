"""Comparison and regression detection."""

from canary.compare.stats import compare_to_baseline
from canary.compare.thresholds import DEFAULT_THRESHOLDS, Thresholds

__all__ = ["compare_to_baseline", "DEFAULT_THRESHOLDS", "Thresholds"]
