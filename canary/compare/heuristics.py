"""Root cause heuristics for regression analysis."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel

from canary.collect.metrics import CanaryMetrics
from canary.compare.stats import CheckStatus, ComparisonReport


class RegressionCategory(str, Enum):
    """Categories of regression root causes."""

    DATALOADER = "dataloader"
    TOKENIZATION = "tokenization"
    COMMUNICATION = "communication"  # NCCL, DDP
    MEMORY = "memory"
    KERNEL = "kernel"  # GPU kernel changes
    IO = "io"  # Checkpointing, logging
    UNKNOWN = "unknown"


class Suspect(BaseModel):
    """A suspected root cause for a regression."""

    category: RegressionCategory
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: list[str]
    suggested_actions: list[str]


class HeuristicAnalysis(BaseModel):
    """Analysis of regression root causes."""

    suspects: list[Suspect]
    summary: str
    top_suspect: Suspect | None = None


def analyze_regression(
    report: ComparisonReport,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> HeuristicAnalysis:
    """Analyze a regression report and suggest root causes.

    This uses heuristics based on which metrics regressed to suggest
    likely root causes.

    Args:
        report: Comparison report showing regressions.
        current: Current run metrics.
        baseline: Baseline run metrics.

    Returns:
        HeuristicAnalysis with suspected root causes.
    """
    suspects: list[Suspect] = []

    # Analyze failed checks
    for check in report.failed_checks:
        suspects.extend(_analyze_check(check, current, baseline))

    # Analyze combined patterns
    suspects.extend(_analyze_patterns(report, current, baseline))

    # Sort by confidence
    suspects.sort(key=lambda s: s.confidence, reverse=True)

    # Generate summary
    if not suspects:
        summary = "No clear regression pattern identified."
    elif len(suspects) == 1:
        summary = f"Most likely cause: {suspects[0].category.value} ({suspects[0].description})"
    else:
        top_categories = [s.category.value for s in suspects[:3]]
        summary = f"Potential causes: {', '.join(top_categories)}"

    return HeuristicAnalysis(
        suspects=suspects,
        summary=summary,
        top_suspect=suspects[0] if suspects else None,
    )


def _analyze_check(
    check: Any,  # Check from stats.py
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze a specific failed check for root causes."""
    suspects = []

    if check.name == "step_time_mean":
        suspects.extend(_analyze_step_time_regression(check, current, baseline))

    elif check.name == "tokens_per_sec":
        suspects.extend(_analyze_throughput_regression(check, current, baseline))

    elif check.name == "max_memory":
        suspects.extend(_analyze_memory_regression(check, current, baseline))

    elif check.name in ("nan_steps", "inf_steps"):
        suspects.extend(_analyze_numerical_instability(check, current, baseline))

    elif check.name == "loss_divergence":
        suspects.extend(_analyze_loss_divergence(check, current, baseline))

    return suspects


def _analyze_step_time_regression(
    check: Any,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze step time regression."""
    suspects = []
    delta_pct = check.delta_pct or 0

    # High step time increase often indicates CPU bottleneck
    if delta_pct > 20:
        suspects.append(
            Suspect(
                category=RegressionCategory.DATALOADER,
                confidence=0.7,
                description="Dataloader or preprocessing bottleneck",
                evidence=[
                    f"Step time increased by {delta_pct:.1f}%",
                    "Large increases often indicate CPU-side bottlenecks",
                ],
                suggested_actions=[
                    "Check dataloader num_workers configuration",
                    "Profile CPU utilization during training",
                    "Check for tokenization changes",
                    "Verify data preprocessing efficiency",
                ],
            )
        )

    # Moderate increase could be kernel changes
    if 10 < delta_pct <= 20:
        suspects.append(
            Suspect(
                category=RegressionCategory.KERNEL,
                confidence=0.5,
                description="Possible GPU kernel efficiency change",
                evidence=[
                    f"Step time increased by {delta_pct:.1f}%",
                    "Moderate increases may indicate kernel changes",
                ],
                suggested_actions=[
                    "Check for model architecture changes",
                    "Review batch size and sequence length settings",
                    "Run PyTorch profiler to identify slow kernels",
                ],
            )
        )

    return suspects


def _analyze_throughput_regression(
    check: Any,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze tokens/sec regression."""
    suspects = []
    delta_pct = check.delta_pct or 0

    # Throughput drop with memory increase suggests memory pressure
    if delta_pct < -10:
        mem_delta = current.perf.max_mem_mb - baseline.perf.max_mem_mb

        if mem_delta > 100:
            suspects.append(
                Suspect(
                    category=RegressionCategory.MEMORY,
                    confidence=0.8,
                    description="Memory pressure causing throughput drop",
                    evidence=[
                        f"Throughput dropped by {-delta_pct:.1f}%",
                        f"Memory increased by {mem_delta:.0f}MB",
                    ],
                    suggested_actions=[
                        "Check for memory leaks",
                        "Review gradient accumulation settings",
                        "Consider gradient checkpointing",
                        "Check CUDA allocator fragmentation",
                    ],
                )
            )
        else:
            suspects.append(
                Suspect(
                    category=RegressionCategory.DATALOADER,
                    confidence=0.6,
                    description="Possible data pipeline bottleneck",
                    evidence=[
                        f"Throughput dropped by {-delta_pct:.1f}%",
                        "Memory stable, suggesting CPU-side issue",
                    ],
                    suggested_actions=[
                        "Profile dataloader wait time",
                        "Check for I/O bottlenecks",
                        "Review data augmentation changes",
                    ],
                )
            )

    return suspects


def _analyze_memory_regression(
    check: Any,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze memory regression."""
    suspects = []
    delta_mb = check.delta or 0

    if delta_mb > 500:
        suspects.append(
            Suspect(
                category=RegressionCategory.MEMORY,
                confidence=0.9,
                description="Significant memory increase",
                evidence=[
                    f"Peak memory increased by {delta_mb:.0f}MB",
                ],
                suggested_actions=[
                    "Check for new model components or parameters",
                    "Review activation checkpointing settings",
                    "Look for memory leaks in callbacks",
                    "Check batch size and sequence length",
                    "Verify gradient accumulation is working",
                ],
            )
        )
    elif delta_mb > 200:
        suspects.append(
            Suspect(
                category=RegressionCategory.MEMORY,
                confidence=0.6,
                description="Moderate memory increase",
                evidence=[
                    f"Peak memory increased by {delta_mb:.0f}MB",
                ],
                suggested_actions=[
                    "Review model configuration changes",
                    "Check for new logging or metrics overhead",
                ],
            )
        )

    return suspects


def _analyze_numerical_instability(
    check: Any,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze NaN/Inf numerical instability."""
    suspects = []

    suspects.append(
        Suspect(
            category=RegressionCategory.UNKNOWN,
            confidence=0.9,
            description="Numerical instability detected",
            evidence=[
                f"NaN steps: {current.stability.nan_steps}",
                f"Inf steps: {current.stability.inf_steps}",
            ],
            suggested_actions=[
                "Check learning rate (may be too high)",
                "Review gradient clipping settings",
                "Check for division by zero in loss computation",
                "Verify input data normalization",
                "Check mixed precision settings (bf16/fp16)",
                "Review model initialization",
            ],
        )
    )

    return suspects


def _analyze_loss_divergence(
    check: Any,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze loss divergence."""
    suspects = []

    suspects.append(
        Suspect(
            category=RegressionCategory.UNKNOWN,
            confidence=0.8,
            description="Training loss diverged",
            evidence=[
                "Loss increased significantly over training",
            ],
            suggested_actions=[
                "Check learning rate schedule",
                "Review optimizer configuration",
                "Verify data quality and ordering",
                "Check for catastrophic forgetting",
                "Review KL/beta settings (for DPO/PPO)",
            ],
        )
    )

    return suspects


def _analyze_grad_norm_patterns(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze gradient norm patterns for instability indicators.

    Uses the collected grad_norm_values to detect:
    - Gradient explosion (increasing trend)
    - High variance (unstable training)
    """
    import statistics

    suspects = []
    grad_norms = current.stability.grad_norm_values

    if len(grad_norms) < 10:
        return suspects

    # Check for gradient explosion (increasing trend)
    third = len(grad_norms) // 3
    if third > 0:
        early_norms = grad_norms[:third]
        late_norms = grad_norms[-third:]

        if early_norms and late_norms:
            early_mean = statistics.mean(early_norms)
            late_mean = statistics.mean(late_norms)

            if early_mean > 0 and late_mean > early_mean * 2:
                increase_pct = (late_mean - early_mean) / early_mean * 100
                suspects.append(
                    Suspect(
                        category=RegressionCategory.UNKNOWN,
                        confidence=0.85,
                        description="Gradient explosion detected",
                        evidence=[
                            f"Early gradient norm avg: {early_mean:.4f}",
                            f"Late gradient norm avg: {late_mean:.4f}",
                            f"Increase: {increase_pct:.1f}%",
                        ],
                        suggested_actions=[
                            "Reduce learning rate",
                            "Increase gradient clipping threshold",
                            "Check for data anomalies in later batches",
                            "Review optimizer settings",
                        ],
                    )
                )

    # Check for gradient variance (unstable training)
    if len(grad_norms) > 20:
        mean = statistics.mean(grad_norms)
        if mean > 0:
            variance = statistics.variance(grad_norms)
            cv = (variance**0.5) / mean  # Coefficient of variation

            if cv > 0.5:  # High variance relative to mean
                suspects.append(
                    Suspect(
                        category=RegressionCategory.UNKNOWN,
                        confidence=0.6,
                        description="Unstable gradient norms",
                        evidence=[
                            f"Gradient norm coefficient of variation: {cv:.2f}",
                            "High variance suggests training instability",
                        ],
                        suggested_actions=[
                            "Consider gradient accumulation",
                            "Check batch size and learning rate",
                            "Review data shuffling and augmentation",
                        ],
                    )
                )

    return suspects


def _analyze_gpu_utilization(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze GPU utilization for bottleneck detection.

    Low GPU utilization suggests CPU-side bottlenecks.
    A significant drop from baseline indicates a new bottleneck.
    """
    suspects = []

    current_gpu = current.perf.gpu_utilization_avg
    baseline_gpu = baseline.perf.gpu_utilization_avg

    if current_gpu is None:
        return suspects

    # Low GPU utilization suggests CPU bottleneck
    if current_gpu < 50:
        evidence = [f"Current GPU utilization: {current_gpu:.1f}%"]
        if baseline_gpu is not None:
            evidence.append(f"Baseline GPU utilization: {baseline_gpu:.1f}%")

        suspects.append(
            Suspect(
                category=RegressionCategory.DATALOADER,
                confidence=0.8,
                description="Low GPU utilization indicates CPU bottleneck",
                evidence=evidence,
                suggested_actions=[
                    "Increase dataloader num_workers",
                    "Enable pin_memory in DataLoader",
                    "Check for CPU-bound preprocessing",
                    "Consider prefetching data",
                ],
            )
        )

    # GPU utilization drop from baseline
    if baseline_gpu is not None and baseline_gpu > 0 and current_gpu < baseline_gpu * 0.8:
        drop_pct = (baseline_gpu - current_gpu) / baseline_gpu * 100
        suspects.append(
            Suspect(
                category=RegressionCategory.DATALOADER,
                confidence=0.7,
                description="GPU utilization dropped significantly",
                evidence=[
                    f"Baseline GPU utilization: {baseline_gpu:.1f}%",
                    f"Current GPU utilization: {current_gpu:.1f}%",
                    f"Drop: {drop_pct:.1f}%",
                ],
                suggested_actions=[
                    "Profile dataloader wait time",
                    "Check for I/O changes",
                    "Review CPU load during training",
                ],
            )
        )

    return suspects


def _analyze_dataloader_wait(
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze dataloader wait time percentage.

    High dataloader wait percentage is a clear signal of data loading bottleneck.
    """
    suspects = []

    current_wait = current.perf.dataloader_wait_pct
    baseline_wait = baseline.perf.dataloader_wait_pct

    if current_wait is None:
        return suspects

    # High dataloader wait time (>20% of step time)
    if current_wait > 20:
        suspects.append(
            Suspect(
                category=RegressionCategory.DATALOADER,
                confidence=0.9,
                description="Significant time spent waiting for data",
                evidence=[
                    f"Dataloader wait time: {current_wait:.1f}% of step time",
                ],
                suggested_actions=[
                    "Increase num_workers in DataLoader",
                    "Enable persistent_workers=True",
                    "Use pin_memory=True",
                    "Consider pre-caching dataset",
                ],
            )
        )

    # Wait time increased significantly from baseline
    if baseline_wait is not None and baseline_wait > 0 and current_wait > baseline_wait * 1.5:
        increase_pct = (current_wait - baseline_wait) / baseline_wait * 100
        suspects.append(
            Suspect(
                category=RegressionCategory.DATALOADER,
                confidence=0.75,
                description="Dataloader wait time increased",
                evidence=[
                    f"Baseline wait: {baseline_wait:.1f}%",
                    f"Current wait: {current_wait:.1f}%",
                    f"Increase: {increase_pct:.1f}%",
                ],
                suggested_actions=[
                    "Check for changes in data preprocessing",
                    "Review dataset loading code",
                    "Profile CPU utilization",
                ],
            )
        )

    return suspects


def _analyze_patterns(
    report: ComparisonReport,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
) -> list[Suspect]:
    """Analyze combined patterns across multiple checks."""
    suspects = []

    failed_names = {c.name for c in report.failed_checks}

    # Pattern: Step time + memory = likely memory fragmentation
    if "step_time_mean" in failed_names and "max_memory" in failed_names:
        suspects.append(
            Suspect(
                category=RegressionCategory.MEMORY,
                confidence=0.75,
                description="Memory fragmentation causing slowdown",
                evidence=[
                    "Both step time and memory regressed",
                    "Pattern suggests CUDA allocator issues",
                ],
                suggested_actions=[
                    "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                    "Consider torch.cuda.memory._set_allocator_settings()",
                    "Review checkpoint frequency",
                ],
            )
        )

    # Pattern: Only step time = likely CPU bottleneck
    if "step_time_mean" in failed_names and "max_memory" not in failed_names:
        if "tokens_per_sec" in failed_names:
            suspects.append(
                Suspect(
                    category=RegressionCategory.DATALOADER,
                    confidence=0.7,
                    description="CPU-side bottleneck (dataloader/tokenization)",
                    evidence=[
                        "Step time regressed without memory increase",
                        "Throughput dropped proportionally",
                    ],
                    suggested_actions=[
                        "Profile CPU utilization",
                        "Check GIL contention",
                        "Review dataloader workers",
                        "Check for Python-side overhead",
                    ],
                )
            )

    # Analyze gradient norm patterns
    suspects.extend(_analyze_grad_norm_patterns(current, baseline))

    # Analyze GPU utilization
    suspects.extend(_analyze_gpu_utilization(current, baseline))

    # Analyze dataloader wait time
    suspects.extend(_analyze_dataloader_wait(current, baseline))

    return suspects


def format_suspects_markdown(analysis: HeuristicAnalysis) -> str:
    """Format suspects as markdown for reporting."""
    if not analysis.suspects:
        return "No clear root cause identified.\n"

    lines = ["## Root Cause Analysis\n"]
    lines.append(f"**Summary:** {analysis.summary}\n")

    for i, suspect in enumerate(analysis.suspects[:3], 1):
        confidence_bar = "█" * int(suspect.confidence * 10) + "░" * (10 - int(suspect.confidence * 10))
        lines.append(f"\n### #{i} {suspect.category.value.title()} ({confidence_bar} {suspect.confidence:.0%})")
        lines.append(f"\n{suspect.description}\n")

        if suspect.evidence:
            lines.append("\n**Evidence:**")
            for ev in suspect.evidence:
                lines.append(f"- {ev}")

        if suspect.suggested_actions:
            lines.append("\n**Suggested Actions:**")
            for action in suspect.suggested_actions:
                lines.append(f"- {action}")

    return "\n".join(lines)
