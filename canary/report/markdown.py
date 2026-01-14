"""Markdown report generation for canary results."""

from __future__ import annotations

from canary.collect.metrics import CanaryMetrics
from canary.compare.heuristics import (
    analyze_regression,
    format_suspects_markdown,
)
from canary.compare.stats import CheckStatus, ComparisonReport


def generate_markdown_report(
    report: ComparisonReport,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    include_heuristics: bool = True,
) -> str:
    """Generate a markdown report from a comparison.

    Args:
        report: Comparison report.
        current: Current run metrics.
        baseline: Baseline run metrics.
        include_heuristics: Whether to include root cause analysis.

    Returns:
        Markdown formatted report string.
    """
    lines: list[str] = []

    # Header with overall status
    status_emoji = "✅" if report.passed else "❌"
    lines.append(f"# RLHF Canary Report {status_emoji}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Status:** {'PASS' if report.passed else 'FAIL'}")
    lines.append(f"**Baseline Run:** `{report.baseline_run_id}`")
    lines.append(f"**Current Run:** `{report.current_run_id}`")
    lines.append("")

    # Environment compatibility warning
    if not report.env_compatible:
        lines.append("⚠️ **Environment Warning:** Hardware differs between runs.")
        for warning in report.env_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    # Checks table
    lines.append("## Regression Checks")
    lines.append("")
    lines.append("| Check | Status | Baseline | Current | Change | Threshold |")
    lines.append("|-------|--------|----------|---------|--------|-----------|")

    for check in report.checks:
        status_icon = _get_status_icon(check.status)
        baseline_val = _format_value(check.baseline_value)
        current_val = _format_value(check.current_value)
        change = _format_change(check.delta, check.delta_pct)
        threshold = _format_value(check.threshold)

        lines.append(
            f"| {check.name} | {status_icon} | {baseline_val} | {current_val} | {change} | {threshold} |"
        )

    lines.append("")

    # Failed checks detail
    if report.failed_checks:
        lines.append("## Failed Checks")
        lines.append("")
        for check in report.failed_checks:
            lines.append(f"### ❌ {check.name}")
            lines.append("")
            lines.append(f"**{check.message}**")
            lines.append("")
            if check.baseline_value is not None:
                lines.append(f"- Baseline: {_format_value(check.baseline_value)}")
            if check.current_value is not None:
                lines.append(f"- Current: {_format_value(check.current_value)}")
            if check.delta_pct is not None:
                lines.append(f"- Change: {check.delta_pct:+.1f}%")
            lines.append("")

    # Warning checks
    if report.warning_checks:
        lines.append("## Warnings")
        lines.append("")
        for check in report.warning_checks:
            lines.append(f"⚠️ **{check.name}:** {check.message}")
        lines.append("")

    # Root cause analysis
    if include_heuristics and not report.passed:
        analysis = analyze_regression(report, current, baseline)
        lines.append(format_suspects_markdown(analysis))
        lines.append("")

    # Performance summary
    lines.append("## Performance Summary")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Change |")
    lines.append("|--------|----------|---------|--------|")

    # Step time
    b_step = baseline.perf.step_time.mean
    c_step = current.perf.step_time.mean
    if b_step and c_step:
        change = ((c_step - b_step) / b_step * 100) if b_step else 0
        lines.append(f"| Step Time (mean) | {b_step:.4f}s | {c_step:.4f}s | {change:+.1f}% |")

    # Tokens/sec
    b_tps = baseline.perf.approx_tokens_per_sec
    c_tps = current.perf.approx_tokens_per_sec
    if b_tps and c_tps:
        change = ((c_tps - b_tps) / b_tps * 100) if b_tps else 0
        lines.append(f"| Tokens/sec | {b_tps:.0f} | {c_tps:.0f} | {change:+.1f}% |")

    # Memory
    b_mem = baseline.perf.max_mem_mb
    c_mem = current.perf.max_mem_mb
    change = ((c_mem - b_mem) / b_mem * 100) if b_mem else 0
    lines.append(f"| Peak Memory | {b_mem:.0f}MB | {c_mem:.0f}MB | {change:+.1f}% |")

    lines.append("")

    # Configuration comparison
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Model:** {current.config.model_name}")
    lines.append(f"- **Max Steps:** {current.config.max_steps}")
    lines.append(f"- **Batch Size:** {current.config.batch_size}")
    lines.append(f"- **Gradient Accumulation:** {current.config.gradient_accumulation_steps}")
    lines.append(f"- **Max Length:** {current.config.max_length}")
    if current.config.beta:
        lines.append(f"- **Beta (DPO):** {current.config.beta}")
    lines.append("")

    return "\n".join(lines)


def generate_short_summary(report: ComparisonReport) -> str:
    """Generate a short one-line summary for CI output.

    Args:
        report: Comparison report.

    Returns:
        Short summary string.
    """
    status = "PASS ✅" if report.passed else "FAIL ❌"
    failed_names = [c.name for c in report.failed_checks]

    if failed_names:
        return f"{status} | Failed: {', '.join(failed_names)}"
    else:
        return status


def _get_status_icon(status: CheckStatus) -> str:
    """Get emoji icon for check status."""
    icons = {
        CheckStatus.PASS: "✅",
        CheckStatus.FAIL: "❌",
        CheckStatus.WARN: "⚠️",
        CheckStatus.SKIP: "⏭️",
    }
    return icons.get(status, "❓")


def _format_value(value: float | None) -> str:
    """Format a numeric value for display."""
    if value is None:
        return "-"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _format_change(delta: float | None, delta_pct: float | None) -> str:
    """Format a change value for display."""
    if delta is None and delta_pct is None:
        return "-"
    if delta_pct is not None:
        return f"{delta_pct:+.1f}%"
    if delta is not None:
        return f"{delta:+.2f}"
    return "-"
