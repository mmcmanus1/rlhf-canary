#!/usr/bin/env python3
"""Comprehensive notebook testing: dry-run execution + static checks."""

import copy
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Patterns for cells to skip (Colab-specific or long-running)
SKIP_PATTERNS = [
    r"!git clone",           # Colab repo setup
    r"%cd /content",         # Colab directory changes
    r"canary\.cli run",      # Training runs via CLI
    r"canary run",           # Training runs
    r"!cp \{",               # Copy metrics files (uses f-string vars)
    r"!mkdir -p baselines",  # Baseline dir creation
    r"shutil\.copy",         # Copy operations
    r"next\(Path.*rglob.*metrics\.json",  # Find metrics files
    r"list\(Path\('\./",     # List files in output directories
    r"Path\('\./profiler_output",  # Profiler output references
    r"Path\('\./stability_output", # Stability output references
    r"Path\('\./rca_output",       # RCA output references
    r"Path\('\./ppo_output",       # PPO output references
    r"Path\('\./sft_output",       # SFT output references
    r"with open\(.*_path\)",       # Open metrics files by path var
    r"compare_to_baseline\(",      # Comparison functions need real data
    r"analyze_regression\(",       # Analysis functions need real data
    # Variables from skipped data-loading cells
    r"profiler_summary",           # From profiler metrics loading
    r"if 'profiler' in metrics",   # References loaded metrics
    r"unstable_paths",             # From stability metrics loading
    r"stable_path",                # From stability baseline
    r"baseline_metrics",           # From baseline loading
    r"current_metrics",            # From current run loading
    r"sft_metrics",                # From SFT loading
    r"dpo_metrics",                # From DPO loading
    r"ppo_metrics",                # From PPO loading
    r"def create_debug_story",     # Helper that needs real data
    r"create_debug_story\(",       # Calls helper
    r"format_suspects_markdown",   # Needs real analysis
    r"def match_regression_pattern",  # Helper that needs real report
    r"match_regression_pattern\(",    # Calls helper
]

# Additional patterns that indicate we should mock the cell
MOCK_PATTERNS = [
    r"Path\('./canary_output",  # References to training output
    r"baselines/.*\.json",      # Baseline file references
]

# Secret patterns to detect
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{20,}", "OpenAI API key"),
    (r"ghp_[a-zA-Z0-9]{36}", "GitHub personal access token"),
    (r"gho_[a-zA-Z0-9]{36}", "GitHub OAuth token"),
    (r"github_pat_[a-zA-Z0-9_]{22,}", "GitHub fine-grained PAT"),
    (r"xox[baprs]-[a-zA-Z0-9-]+", "Slack token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
    (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
    (r"token\s*=\s*['\"][a-zA-Z0-9_-]{20,}['\"]", "Hardcoded token"),
    (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
]

# Output size limit (100KB)
MAX_OUTPUT_SIZE_BYTES = 100 * 1024

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    message: str = ""
    details: list[str] = field(default_factory=list)


@dataclass
class NotebookResult:
    """All check results for a notebook."""
    name: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def check_structure(nb: nbformat.NotebookNode) -> CheckResult:
    """Validate notebook structure with nbformat."""
    try:
        nbformat.validate(nb)
        return CheckResult("structure", True, "Valid notebook format")
    except nbformat.ValidationError as e:
        return CheckResult("structure", False, f"Invalid: {e.message[:100]}")


def check_secrets(nb: nbformat.NotebookNode) -> CheckResult:
    """Scan for hardcoded secrets in code cells."""
    found = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        for pattern, description in SECRET_PATTERNS:
            if re.search(pattern, source, re.IGNORECASE):
                found.append(f"Cell {i+1}: {description}")

    if found:
        return CheckResult("secrets", False, f"Found {len(found)} potential secrets", found)
    return CheckResult("secrets", True, "No secrets detected")


def check_outputs(nb: nbformat.NotebookNode) -> CheckResult:
    """Check for oversized cell outputs."""
    large_outputs = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        outputs = cell.get("outputs", [])
        for j, output in enumerate(outputs):
            # Estimate output size
            output_str = json.dumps(output, default=str)
            size = len(output_str.encode("utf-8"))

            if size > MAX_OUTPUT_SIZE_BYTES:
                size_kb = size / 1024
                large_outputs.append(f"Cell {i+1}, output {j+1}: {size_kb:.1f}KB")

    if large_outputs:
        return CheckResult(
            "outputs",
            False,
            f"Found {len(large_outputs)} large outputs (>100KB)",
            large_outputs
        )
    return CheckResult("outputs", True, "No oversized outputs")


def check_execution_order(nb: nbformat.NotebookNode) -> CheckResult:
    """Check for out-of-order execution."""
    counts = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        exec_count = cell.get("execution_count")
        if exec_count is not None:
            counts.append((i, exec_count))

    if not counts:
        return CheckResult("exec_order", True, "No execution counts (fresh notebook)")

    # Check if execution counts are sequential
    issues = []
    prev_count = 0
    for cell_idx, exec_count in counts:
        if exec_count < prev_count:
            issues.append(f"Cell {cell_idx+1}: executed out of order (count {exec_count})")
        prev_count = exec_count

    if issues:
        return CheckResult("exec_order", False, "Out-of-order execution detected", issues)
    return CheckResult("exec_order", True, "Execution order OK")


def check_lint(nb: nbformat.NotebookNode, notebook_name: str) -> CheckResult:
    """Lint code cells with ruff."""
    # Combine all code cells into a single file for linting
    code_lines = []

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        source = cell.source
        # Skip cells that would be skipped in execution
        if any(re.search(p, source) for p in SKIP_PATTERNS):
            continue

        code_lines.append(f"# Cell {i+1}")
        code_lines.append(source)
        code_lines.append("")  # blank line between cells

    if not code_lines:
        return CheckResult("lint", True, "No code to lint")

    combined_code = "\n".join(code_lines)

    # Write to temp file and run ruff
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(combined_code)
        temp_path = f.name

    try:
        # Ignore: F401 (unused imports - common in notebook setup cells),
        # E501 (line too long - notebooks often have verbose output),
        # F841 (unused variables - notebooks use intermediate results for display)
        result = subprocess.run(
            ["ruff", "check", temp_path, "--select=E,F", "--ignore=F401,E501,F841"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return CheckResult("lint", True, "No lint errors")

        # Parse ruff output
        issues = []
        for line in result.stdout.strip().split("\n"):
            if line and ":" in line:
                # Extract just the error part
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append(parts[3].strip()[:60])

        if issues:
            return CheckResult("lint", False, f"Found {len(issues)} lint issues", issues[:5])
        return CheckResult("lint", True, "No lint errors")

    finally:
        Path(temp_path).unlink(missing_ok=True)


def should_skip_cell(source: str) -> bool:
    """Check if a cell should be skipped based on its content."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, source):
            return True
    return False


def should_mock_cell(source: str) -> bool:
    """Check if a cell references training outputs that won't exist."""
    for pattern in MOCK_PATTERNS:
        if re.search(pattern, source):
            return True
    return False


def filter_notebook(nb: nbformat.NotebookNode) -> tuple[nbformat.NotebookNode, int, int]:
    """Filter notebook cells, replacing skipped cells with pass statements."""
    skipped = 0
    mocked = 0

    for cell in nb.cells:
        if cell.cell_type != "code":
            continue

        source = cell.source

        if should_skip_cell(source):
            cell.source = "# [SKIPPED] " + source.split("\n")[0][:50] + "..."
            skipped += 1
        elif should_mock_cell(source):
            cell.source = "# [MOCKED - references training output]\npass"
            mocked += 1

    return nb, skipped, mocked


def check_execution(nb: nbformat.NotebookNode, notebook_path: Path) -> CheckResult:
    """Execute notebook cells (with filtering)."""
    nb_copy = copy.deepcopy(nb)
    total_code_cells = sum(1 for c in nb_copy.cells if c.cell_type == "code")

    nb_copy, skipped, mocked = filter_notebook(nb_copy)
    executed = total_code_cells - skipped - mocked

    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    project_root = notebook_path.parent.parent

    try:
        ep.preprocess(nb_copy, {"metadata": {"path": str(project_root)}})
        return CheckResult(
            "execution",
            True,
            f"Executed {executed}/{total_code_cells} cells (skipped {skipped}, mocked {mocked})"
        )
    except Exception as e:
        error_msg = str(e)
        if "CellExecutionError" in error_msg:
            lines = error_msg.split("\n")
            for i, line in enumerate(lines):
                if "Error" in line or "Exception" in line:
                    error_msg = lines[i][:100]
                    break
        return CheckResult("execution", False, f"Execution failed: {error_msg[:100]}")


def test_notebook(notebook_path: Path) -> NotebookResult:
    """Run all checks on a notebook."""
    result = NotebookResult(notebook_path.name)

    try:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        result.checks.append(CheckResult("load", False, f"Failed to load: {e}"))
        return result

    # Run all checks
    result.checks.append(check_structure(nb))
    result.checks.append(check_secrets(nb))
    result.checks.append(check_outputs(nb))
    result.checks.append(check_execution_order(nb))
    result.checks.append(check_lint(nb, notebook_path.name))
    result.checks.append(check_execution(nb, notebook_path))

    return result


def main():
    """Run tests on all notebooks."""
    print("=" * 70)
    print("RLHF Canary Notebook Test Suite")
    print("=" * 70)

    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))

    if not notebooks:
        print(f"No notebooks found in {NOTEBOOKS_DIR}")
        sys.exit(1)

    print(f"Found {len(notebooks)} notebooks\n")
    print("Checks: structure, secrets, outputs, exec_order, lint, execution\n")

    results = []
    for nb_path in notebooks:
        print(f"\n{'-'*70}")
        print(f"Testing: {nb_path.name}")
        print(f"{'-'*70}")

        result = test_notebook(nb_path)
        results.append(result)

        for check in result.checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.name}: {check.message}")
            for detail in check.details[:3]:
                print(f"      - {detail}")
            if len(check.details) > 3:
                print(f"      ... and {len(check.details) - 3} more")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    for result in results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"  {status}: {result.name}")
        for check in result.failed_checks:
            print(f"         - {check.name}: {check.message[:50]}")

    print(f"\nTotal: {passed}/{len(results)} notebooks passed all checks")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
