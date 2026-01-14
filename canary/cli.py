"""Command-line interface for RLHF Canary."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from canary import __version__


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """RLHF Canary - Regression detection for RLHF/finetuning pipelines."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Override output directory from config",
)
@click.pass_context
def run(ctx: click.Context, config_path: str, output_dir: str | None) -> None:
    """Run a canary training job.

    CONFIG_PATH: Path to the YAML configuration file.
    """
    from canary.runner.base import load_config
    from canary.runner.local import LocalRunner

    click.echo(f"Loading config from: {config_path}")
    config = load_config(config_path)

    if output_dir:
        config.output_dir = output_dir

    click.echo(f"Starting canary run: {config.name}")
    click.echo(f"  Model: {config.model_name}")
    click.echo(f"  Training type: {config.training_type}")
    click.echo(f"  Max steps: {config.max_steps}")

    runner = LocalRunner(config)
    result = runner.run()

    if result.success:
        click.echo(click.style("\n✅ Canary run completed successfully!", fg="green"))
        click.echo(f"Metrics saved to: {result.metrics_path}")

        if result.metrics:
            click.echo("\nQuick Summary:")
            click.echo(f"  Duration: {result.metrics.duration_seconds:.1f}s")
            if result.metrics.perf.step_time.mean:
                click.echo(f"  Step time (mean): {result.metrics.perf.step_time.mean:.4f}s")
            if result.metrics.perf.approx_tokens_per_sec:
                click.echo(f"  Tokens/sec: {result.metrics.perf.approx_tokens_per_sec:.0f}")
            click.echo(f"  Peak memory: {result.metrics.perf.max_mem_mb:.0f}MB")
            click.echo(f"  NaN steps: {result.metrics.stability.nan_steps}")
    else:
        click.echo(click.style(f"\n❌ Canary run failed: {result.error}", fg="red"))
        sys.exit(1)


@main.command()
@click.argument("current_path", type=click.Path(exists=True))
@click.argument("baseline_path", type=click.Path(exists=True))
@click.option(
    "--threshold-tier",
    type=click.Choice(["default", "smoke", "perf", "nightly"]),
    default="default",
    help="Threshold tier to use (ignored if --threshold-file provided)",
)
@click.option(
    "--threshold-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom threshold YAML file (overrides --threshold-tier)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for comparison report",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def compare(
    ctx: click.Context,
    current_path: str,
    baseline_path: str,
    threshold_tier: str,
    threshold_file: str | None,
    output: str | None,
    json_output: bool,
) -> None:
    """Compare current metrics to baseline.

    CURRENT_PATH: Path to current run metrics.json
    BASELINE_PATH: Path to baseline metrics.json
    """
    from canary.compare.stats import compare_to_baseline, load_metrics
    from canary.compare.thresholds import get_thresholds, load_thresholds_from_yaml
    from canary.report.markdown import generate_markdown_report

    click.echo(f"Loading current metrics: {current_path}")
    current = load_metrics(current_path)

    click.echo(f"Loading baseline metrics: {baseline_path}")
    baseline = load_metrics(baseline_path)

    # Load thresholds from file or use tier
    if threshold_file:
        click.echo(f"Loading custom thresholds from: {threshold_file}")
        thresholds = load_thresholds_from_yaml(threshold_file)
    else:
        click.echo(f"Using threshold tier: {threshold_tier}")
        thresholds = get_thresholds(threshold_tier)

    report = compare_to_baseline(current, baseline, thresholds)

    if json_output:
        output_data = report.model_dump_json(indent=2)
        if output:
            Path(output).write_text(output_data)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(output_data)
    else:
        markdown = generate_markdown_report(report, current, baseline)
        if output:
            Path(output).write_text(markdown)
            click.echo(f"Report saved to: {output}")
        else:
            click.echo(markdown)

    # Exit with status code
    if report.passed:
        click.echo(click.style("\n✅ All checks passed!", fg="green"))
    else:
        click.echo(click.style("\n❌ Regression detected!", fg="red"))
        for check in report.failed_checks:
            click.echo(f"  - {check.name}: {check.message}")
        sys.exit(1)


@main.command("save-baseline")
@click.argument("metrics_path", type=click.Path(exists=True))
@click.argument("baseline_path", type=click.Path())
@click.pass_context
def save_baseline(
    ctx: click.Context,
    metrics_path: str,
    baseline_path: str,
) -> None:
    """Save metrics as a new baseline.

    METRICS_PATH: Path to the metrics.json to save as baseline
    BASELINE_PATH: Path where baseline should be saved
    """
    import shutil

    metrics_path = Path(metrics_path)
    baseline_path = Path(baseline_path)

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(metrics_path, baseline_path)

    click.echo(f"Saved baseline to: {baseline_path}")


@main.command()
@click.argument("comparison_path", type=click.Path(exists=True))
@click.option("--repo", help="GitHub repository (owner/repo)")
@click.option("--pr", type=int, help="PR number to comment on")
@click.pass_context
def report(
    ctx: click.Context,
    comparison_path: str,
    repo: str | None,
    pr: int | None,
) -> None:
    """Post comparison report to GitHub PR.

    COMPARISON_PATH: Path to comparison report JSON
    """
    from canary.compare.stats import ComparisonReport, load_metrics
    from canary.report.github import post_github_comment

    click.echo(f"Loading comparison report: {comparison_path}")

    # For now, we need the full metrics to generate the report
    # In a real implementation, we'd save the full report or pass metrics paths
    click.echo("Note: This command requires metrics files. Use --help for details.")
    click.echo("For now, use the compare command to generate reports.")


@main.command()
@click.pass_context
def env(ctx: click.Context) -> None:
    """Show environment fingerprint."""
    from canary.collect.env_fingerprint import get_env_fingerprint

    fingerprint = get_env_fingerprint()

    click.echo("Environment Fingerprint:")
    click.echo(f"  Python: {fingerprint.python_version}")
    click.echo(f"  PyTorch: {fingerprint.torch_version}")
    click.echo(f"  CUDA available: {fingerprint.cuda_available}")
    if fingerprint.cuda_available:
        click.echo(f"  CUDA version: {fingerprint.cuda_version}")
        click.echo(f"  GPU: {fingerprint.gpu_name}")
        click.echo(f"  GPU count: {fingerprint.gpu_count}")
        if fingerprint.gpu_memory_gb:
            click.echo(f"  GPU memory: {fingerprint.gpu_memory_gb:.1f}GB")
    click.echo(f"  Platform: {fingerprint.platform} {fingerprint.platform_version}")
    click.echo(f"  Transformers: {fingerprint.transformers_version}")
    click.echo(f"  TRL: {fingerprint.trl_version}")
    click.echo(f"  Fingerprint hash: {fingerprint.fingerprint_hash}")


@main.command("gh-report")
@click.argument("current_path", type=click.Path(exists=True))
@click.argument("baseline_path", type=click.Path(exists=True))
@click.option(
    "--threshold-tier",
    type=click.Choice(["default", "smoke", "perf", "nightly"]),
    default="default",
    help="Threshold tier to use (ignored if --threshold-file provided)",
)
@click.option(
    "--threshold-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to custom threshold YAML file (overrides --threshold-tier)",
)
@click.option("--post-comment/--no-comment", default=True, help="Post PR comment")
@click.option("--update-status/--no-status", default=True, help="Update commit status")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for comparison report",
)
@click.pass_context
def gh_report(
    ctx: click.Context,
    current_path: str,
    baseline_path: str,
    threshold_tier: str,
    threshold_file: str | None,
    post_comment: bool,
    update_status: bool,
    output: str | None,
) -> None:
    """Post comparison results to GitHub PR.

    Runs comparison and posts results as PR comment and commit status.

    CURRENT_PATH: Path to current run metrics.json
    BASELINE_PATH: Path to baseline metrics.json
    """
    from canary.compare.stats import compare_to_baseline, load_metrics
    from canary.compare.thresholds import get_thresholds, load_thresholds_from_yaml
    from canary.report.github import (
        post_github_comment,
        update_pr_status,
        write_github_summary,
    )
    from canary.report.markdown import generate_markdown_report

    click.echo(f"Loading current metrics: {current_path}")
    current = load_metrics(current_path)

    click.echo(f"Loading baseline metrics: {baseline_path}")
    baseline = load_metrics(baseline_path)

    # Load thresholds from file or use tier
    if threshold_file:
        click.echo(f"Loading custom thresholds from: {threshold_file}")
        thresholds = load_thresholds_from_yaml(threshold_file)
    else:
        click.echo(f"Using threshold tier: {threshold_tier}")
        thresholds = get_thresholds(threshold_tier)

    report = compare_to_baseline(current, baseline, thresholds)

    # Generate markdown report
    markdown = generate_markdown_report(report, current, baseline)

    # Save report if output path provided
    if output:
        Path(output).write_text(markdown)
        click.echo(f"Report saved to: {output}")

    # Write to GitHub job summary
    write_github_summary(markdown)

    # Post PR comment
    if post_comment:
        click.echo("Posting PR comment...")
        if post_github_comment(report, current, baseline):
            click.echo(click.style("✓ Posted PR comment", fg="green"))
        else:
            click.echo(click.style("⚠ Failed to post PR comment", fg="yellow"))

    # Update commit status
    if update_status:
        click.echo("Updating commit status...")
        if update_pr_status(report):
            click.echo(click.style("✓ Updated commit status", fg="green"))
        else:
            click.echo(click.style("⚠ Failed to update commit status", fg="yellow"))

    # Exit with status code
    if report.passed:
        click.echo(click.style("\n✅ All checks passed!", fg="green"))
    else:
        click.echo(click.style("\n❌ Regression detected!", fg="red"))
        for check in report.failed_checks:
            click.echo(f"  - {check.name}: {check.message}")
        sys.exit(1)


@main.command("init-config")
@click.argument("output_path", type=click.Path())
@click.option(
    "--type",
    "training_type",
    type=click.Choice(["dpo", "sft"]),
    default="dpo",
    help="Training type",
)
@click.option(
    "--tier",
    type=click.Choice(["smoke", "perf", "nightly"]),
    default="smoke",
    help="Test tier (affects step count)",
)
@click.pass_context
def init_config(
    ctx: click.Context,
    output_path: str,
    training_type: str,
    tier: str,
) -> None:
    """Generate a sample configuration file.

    OUTPUT_PATH: Where to save the config file
    """
    import yaml

    # Step counts by tier
    steps = {"smoke": 100, "perf": 500, "nightly": 2000}

    config = {
        "name": f"{training_type}_{tier}",
        "description": f"{training_type.upper()} {tier} canary test",
        "model_name": "EleutherAI/pythia-70m",
        "use_peft": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "training_type": training_type,
        "max_steps": steps[tier],
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "max_length": 256,
        "warmup_steps": 10,
        "dataset_name": "Anthropic/hh-rlhf",
        "dataset_split": "train",
        "dataset_size": 1024,
        "seed": 42,
        "output_dir": "./canary_output",
    }

    if training_type == "dpo":
        config["beta"] = 0.1
        config["max_prompt_length"] = 64

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.echo(f"Created config: {output_path}")
    click.echo(f"  Type: {training_type}")
    click.echo(f"  Tier: {tier}")
    click.echo(f"  Steps: {steps[tier]}")


if __name__ == "__main__":
    main()
