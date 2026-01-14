"""Statistical comparison for regression detection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from canary.collect.env_fingerprint import EnvFingerprint, fingerprints_compatible
from canary.collect.metrics import CanaryMetrics
from canary.compare.thresholds import DEFAULT_THRESHOLDS, Thresholds


class CheckStatus(str, Enum):
    """Status of an individual check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class Check(BaseModel):
    """Result of a single regression check."""

    name: str
    status: CheckStatus
    message: str
    baseline_value: float | None = None
    current_value: float | None = None
    threshold: float | None = None
    delta: float | None = None
    delta_pct: float | None = None


class ComparisonReport(BaseModel):
    """Complete comparison report between baseline and current run."""

    passed: bool
    baseline_run_id: str
    current_run_id: str
    checks: list[Check]
    warnings: list[str]
    env_compatible: bool
    env_warnings: list[str]

    @property
    def failed_checks(self) -> list[Check]:
        """Get list of failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warning_checks(self) -> list[Check]:
        """Get list of warning checks."""
        return [c for c in self.checks if c.status == CheckStatus.WARN]


def _compute_pct_change(baseline: float, current: float) -> float:
    """Compute percentage change from baseline to current."""
    if baseline == 0:
        return 0.0 if current == 0 else float("inf")
    return 100.0 * (current - baseline) / baseline


def compare_to_baseline(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds | None = None,
) -> ComparisonReport:
    """Compare current metrics to baseline and detect regressions.

    Args:
        current: Metrics from the current run.
        baseline: Metrics from the baseline run.
        thresholds: Thresholds for regression detection.

    Returns:
        ComparisonReport with pass/fail status and detailed checks.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    checks: list[Check] = []
    warnings: list[str] = []

    # Check environment compatibility
    baseline_env = EnvFingerprint(**baseline.env)
    current_env = EnvFingerprint(**current.env)
    env_compatible, env_warnings = fingerprints_compatible(baseline_env, current_env)

    # Check: NaN steps
    nan_check = _check_nan_steps(current, baseline, thresholds)
    checks.append(nan_check)

    # Check: Inf steps
    inf_check = _check_inf_steps(current, baseline, thresholds)
    checks.append(inf_check)

    # Check: Step time regression
    step_time_check = _check_step_time(current, baseline, thresholds, env_compatible)
    checks.append(step_time_check)

    # Check: Tokens per second regression
    tps_check = _check_tokens_per_sec(current, baseline, thresholds, env_compatible)
    checks.append(tps_check)

    # Check: Memory regression
    mem_check = _check_memory(current, baseline, thresholds, env_compatible)
    checks.append(mem_check)

    # Check: Loss divergence
    loss_check = _check_loss_divergence(current, baseline, thresholds)
    checks.append(loss_check)

    # Check: Final loss regression
    final_loss_check = _check_final_loss(current, baseline, thresholds)
    checks.append(final_loss_check)

    # Determine overall pass/fail
    passed = all(c.status != CheckStatus.FAIL for c in checks)

    return ComparisonReport(
        passed=passed,
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        checks=checks,
        warnings=warnings,
        env_compatible=env_compatible,
        env_warnings=env_warnings,
    )


def _check_nan_steps(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
) -> Check:
    """Check for NaN steps in the current run."""
    nan_steps = current.stability.nan_steps
    allowed = thresholds.nan_steps_allowed

    if nan_steps > allowed:
        return Check(
            name="nan_steps",
            status=CheckStatus.FAIL,
            message=f"NaN detected in {nan_steps} steps (allowed: {allowed})",
            current_value=nan_steps,
            threshold=allowed,
        )

    return Check(
        name="nan_steps",
        status=CheckStatus.PASS,
        message="No NaN values detected",
        current_value=nan_steps,
        threshold=allowed,
    )


def _check_inf_steps(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
) -> Check:
    """Check for Inf steps in the current run."""
    inf_steps = current.stability.inf_steps
    allowed = thresholds.inf_steps_allowed

    if inf_steps > allowed:
        return Check(
            name="inf_steps",
            status=CheckStatus.FAIL,
            message=f"Inf detected in {inf_steps} steps (allowed: {allowed})",
            current_value=inf_steps,
            threshold=allowed,
        )

    return Check(
        name="inf_steps",
        status=CheckStatus.PASS,
        message="No Inf values detected",
        current_value=inf_steps,
        threshold=allowed,
    )


def _check_step_time(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
    env_compatible: bool,
) -> Check:
    """Check for step time regression."""
    baseline_mean = baseline.perf.step_time.mean
    current_mean = current.perf.step_time.mean

    # Skip if insufficient data
    if baseline_mean is None or current_mean is None:
        return Check(
            name="step_time_mean",
            status=CheckStatus.SKIP,
            message="Insufficient step time data",
        )

    if baseline.perf.step_time.count < thresholds.min_step_count:
        return Check(
            name="step_time_mean",
            status=CheckStatus.SKIP,
            message=f"Baseline has insufficient steps ({baseline.perf.step_time.count} < {thresholds.min_step_count})",
        )

    delta = current_mean - baseline_mean
    delta_pct = _compute_pct_change(baseline_mean, current_mean)

    # Only fail if environments are compatible
    if delta_pct > thresholds.max_step_time_increase_pct:
        status = CheckStatus.FAIL if env_compatible else CheckStatus.WARN
        return Check(
            name="step_time_mean",
            status=status,
            message=f"Step time increased by {delta_pct:.1f}% (threshold: {thresholds.max_step_time_increase_pct}%)",
            baseline_value=baseline_mean,
            current_value=current_mean,
            threshold=thresholds.max_step_time_increase_pct,
            delta=delta,
            delta_pct=delta_pct,
        )

    return Check(
        name="step_time_mean",
        status=CheckStatus.PASS,
        message=f"Step time change: {delta_pct:+.1f}%",
        baseline_value=baseline_mean,
        current_value=current_mean,
        threshold=thresholds.max_step_time_increase_pct,
        delta=delta,
        delta_pct=delta_pct,
    )


def _check_tokens_per_sec(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
    env_compatible: bool,
) -> Check:
    """Check for tokens per second regression."""
    baseline_tps = baseline.perf.approx_tokens_per_sec
    current_tps = current.perf.approx_tokens_per_sec

    if baseline_tps is None or current_tps is None:
        return Check(
            name="tokens_per_sec",
            status=CheckStatus.SKIP,
            message="Insufficient tokens/sec data",
        )

    # For TPS, a decrease is bad (negative change means regression)
    delta = current_tps - baseline_tps
    delta_pct = _compute_pct_change(baseline_tps, current_tps)
    drop_pct = -delta_pct  # Positive drop means regression

    if drop_pct > thresholds.max_tps_drop_pct:
        status = CheckStatus.FAIL if env_compatible else CheckStatus.WARN
        return Check(
            name="tokens_per_sec",
            status=status,
            message=f"Tokens/sec dropped by {drop_pct:.1f}% (threshold: {thresholds.max_tps_drop_pct}%)",
            baseline_value=baseline_tps,
            current_value=current_tps,
            threshold=thresholds.max_tps_drop_pct,
            delta=delta,
            delta_pct=delta_pct,
        )

    return Check(
        name="tokens_per_sec",
        status=CheckStatus.PASS,
        message=f"Tokens/sec change: {delta_pct:+.1f}%",
        baseline_value=baseline_tps,
        current_value=current_tps,
        threshold=thresholds.max_tps_drop_pct,
        delta=delta,
        delta_pct=delta_pct,
    )


def _check_memory(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
    env_compatible: bool,
) -> Check:
    """Check for memory regression."""
    baseline_mem = baseline.perf.max_mem_mb
    current_mem = current.perf.max_mem_mb

    delta = current_mem - baseline_mem
    delta_pct = _compute_pct_change(baseline_mem, current_mem) if baseline_mem > 0 else 0

    # Check both absolute and percentage thresholds
    abs_exceeded = delta > thresholds.max_mem_increase_mb
    pct_exceeded = delta_pct > thresholds.max_mem_increase_pct

    if abs_exceeded or pct_exceeded:
        status = CheckStatus.FAIL if env_compatible else CheckStatus.WARN
        msg_parts = []
        if abs_exceeded:
            msg_parts.append(f"+{delta:.0f}MB > {thresholds.max_mem_increase_mb}MB")
        if pct_exceeded:
            msg_parts.append(f"+{delta_pct:.1f}% > {thresholds.max_mem_increase_pct}%")

        return Check(
            name="max_memory",
            status=status,
            message=f"Memory regression: {'; '.join(msg_parts)}",
            baseline_value=baseline_mem,
            current_value=current_mem,
            threshold=thresholds.max_mem_increase_mb,
            delta=delta,
            delta_pct=delta_pct,
        )

    return Check(
        name="max_memory",
        status=CheckStatus.PASS,
        message=f"Memory change: {delta:+.0f}MB ({delta_pct:+.1f}%)",
        baseline_value=baseline_mem,
        current_value=current_mem,
        threshold=thresholds.max_mem_increase_mb,
        delta=delta,
        delta_pct=delta_pct,
    )


def _check_loss_divergence(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
) -> Check:
    """Check if loss diverged during training."""
    if current.stability.loss_diverged:
        return Check(
            name="loss_divergence",
            status=CheckStatus.FAIL,
            message="Loss diverged during training (late loss >> early loss)",
        )

    return Check(
        name="loss_divergence",
        status=CheckStatus.PASS,
        message="Loss stable during training",
    )


def _check_final_loss(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    thresholds: Thresholds,
) -> Check:
    """Check for final loss regression."""
    baseline_loss = baseline.stability.final_loss
    current_loss = current.stability.final_loss

    if baseline_loss is None or current_loss is None:
        return Check(
            name="final_loss",
            status=CheckStatus.SKIP,
            message="Insufficient loss data",
        )

    delta = current_loss - baseline_loss
    delta_pct = _compute_pct_change(baseline_loss, current_loss)

    if delta_pct > thresholds.max_loss_increase_pct:
        return Check(
            name="final_loss",
            status=CheckStatus.WARN,  # Warn only, as loss can vary
            message=f"Final loss increased by {delta_pct:.1f}% (threshold: {thresholds.max_loss_increase_pct}%)",
            baseline_value=baseline_loss,
            current_value=current_loss,
            threshold=thresholds.max_loss_increase_pct,
            delta=delta,
            delta_pct=delta_pct,
        )

    return Check(
        name="final_loss",
        status=CheckStatus.PASS,
        message=f"Final loss change: {delta_pct:+.1f}%",
        baseline_value=baseline_loss,
        current_value=current_loss,
        threshold=thresholds.max_loss_increase_pct,
        delta=delta,
        delta_pct=delta_pct,
    )


def load_metrics(path: str) -> CanaryMetrics:
    """Load metrics from a JSON file.

    Args:
        path: Path to the metrics JSON file.

    Returns:
        CanaryMetrics loaded from the file.
    """
    import json
    from pathlib import Path

    with open(Path(path)) as f:
        data = json.load(f)

    return CanaryMetrics(**data)
