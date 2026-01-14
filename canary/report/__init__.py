"""Report generation for canary results."""

from canary.report.github import post_github_comment
from canary.report.markdown import generate_markdown_report

__all__ = ["generate_markdown_report", "post_github_comment"]
