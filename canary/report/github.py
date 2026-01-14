"""GitHub integration for posting canary results."""

from __future__ import annotations

import json
import logging
import os
import subprocess

from canary.collect.metrics import CanaryMetrics
from canary.compare.stats import ComparisonReport
from canary.report.markdown import generate_markdown_report, generate_short_summary

logger = logging.getLogger(__name__)


def post_github_comment(
    report: ComparisonReport,
    current: CanaryMetrics,
    baseline: CanaryMetrics,
    repo: str | None = None,
    pr_number: int | None = None,
) -> bool:
    """Post canary results as a GitHub PR comment.

    Requires `gh` CLI to be installed and authenticated.

    Args:
        report: Comparison report.
        current: Current run metrics.
        baseline: Baseline run metrics.
        repo: Repository in "owner/repo" format. Auto-detected if not provided.
        pr_number: PR number to comment on. Auto-detected if not provided.

    Returns:
        True if comment was posted successfully.
    """
    # Generate markdown report
    markdown = generate_markdown_report(report, current, baseline)

    # Add collapsible details
    full_report = f"""
<details>
<summary>🔍 RLHF Canary Results: {generate_short_summary(report)}</summary>

{markdown}

</details>
"""

    # Try to detect repo and PR if not provided
    if repo is None:
        repo = _detect_repo()
    if pr_number is None:
        pr_number = _detect_pr_number()

    if repo is None or pr_number is None:
        logger.error("Could not detect repo or PR number")
        return False

    # Post comment using gh CLI
    try:
        subprocess.run(
            [
                "gh",
                "pr",
                "comment",
                str(pr_number),
                "--repo",
                repo,
                "--body",
                full_report,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Posted comment to {repo}#{pr_number}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to post comment: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("gh CLI not found. Install it from https://cli.github.com/")
        return False


def update_pr_status(
    report: ComparisonReport,
    repo: str | None = None,
    sha: str | None = None,
    context: str = "canary/regression",
) -> bool:
    """Update PR commit status based on canary results.

    Args:
        report: Comparison report.
        repo: Repository in "owner/repo" format.
        sha: Commit SHA to update status for.
        context: Status context name.

    Returns:
        True if status was updated successfully.
    """
    if repo is None:
        repo = _detect_repo()
    if sha is None:
        sha = _detect_commit_sha()

    if repo is None or sha is None:
        logger.error("Could not detect repo or commit SHA")
        return False

    state = "success" if report.passed else "failure"
    description = generate_short_summary(report)[:140]  # GitHub limit

    try:
        subprocess.run(
            [
                "gh",
                "api",
                f"repos/{repo}/statuses/{sha}",
                "-X",
                "POST",
                "-f",
                f"state={state}",
                "-f",
                f"context={context}",
                "-f",
                f"description={description}",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Updated status for {sha[:8]} to {state}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update status: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("gh CLI not found")
        return False


def _detect_repo() -> str | None:
    """Detect repository from git remote or environment."""
    # Check GitHub Actions environment
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    if github_repo:
        return github_repo

    # Try to get from git remote
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()

        # Parse SSH format: git@github.com:owner/repo.git
        if url.startswith("git@github.com:"):
            return url.split(":")[1].replace(".git", "")

        # Parse HTTPS format: https://github.com/owner/repo.git
        if "github.com" in url:
            parts = url.split("github.com/")[1].replace(".git", "")
            return parts

    except (subprocess.CalledProcessError, IndexError):
        pass

    return None


def _detect_pr_number() -> int | None:
    """Detect PR number from environment or git."""
    # GitHub Actions environment
    pr_ref = os.environ.get("GITHUB_REF")
    if pr_ref and pr_ref.startswith("refs/pull/"):
        try:
            return int(pr_ref.split("/")[2])
        except (IndexError, ValueError):
            pass

    # Try gh CLI
    try:
        result = subprocess.run(
            ["gh", "pr", "view", "--json", "number"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return data.get("number")

    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        pass

    return None


def _detect_commit_sha() -> str | None:
    """Detect current commit SHA."""
    # GitHub Actions environment
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha

    # Try git
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError:
        pass

    return None


def set_github_output(key: str, value: str) -> None:
    """Set a GitHub Actions output variable.

    Args:
        key: Output variable name.
        value: Output value.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Log for local testing (GITHUB_OUTPUT not available outside CI)
        logger.info(f"GitHub output (not in CI): {key}={value}")


def write_github_summary(markdown: str) -> None:
    """Write to GitHub Actions job summary.

    Args:
        markdown: Markdown content to add to summary.
    """
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(markdown)
            f.write("\n")
