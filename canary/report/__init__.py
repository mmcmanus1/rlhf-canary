"""Report generation for canary results."""

from canary.report.markdown import generate_markdown_report
from canary.report.github import post_github_comment

__all__ = ["generate_markdown_report", "post_github_comment"]
