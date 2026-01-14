"""Comparison and regression detection."""

from canary.compare.heuristics import (
    HeuristicAnalysis,
    RegressionCategory,
    Suspect,
    analyze_regression,
    format_suspects_markdown,
)
from canary.compare.stats import (
    Check,
    CheckStatus,
    ComparisonReport,
    compare_to_baseline,
    load_metrics,
)
from canary.compare.thresholds import (
    DEFAULT_THRESHOLDS,
    NIGHTLY_THRESHOLDS,
    PERF_THRESHOLDS,
    SMOKE_THRESHOLDS,
    Thresholds,
    get_thresholds,
)

__all__ = [
    # stats
    "compare_to_baseline",
    "load_metrics",
    "CheckStatus",
    "Check",
    "ComparisonReport",
    # thresholds
    "Thresholds",
    "DEFAULT_THRESHOLDS",
    "SMOKE_THRESHOLDS",
    "PERF_THRESHOLDS",
    "NIGHTLY_THRESHOLDS",
    "get_thresholds",
    # heuristics
    "analyze_regression",
    "format_suspects_markdown",
    "HeuristicAnalysis",
    "Suspect",
    "RegressionCategory",
]
