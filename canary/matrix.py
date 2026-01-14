"""Test matrix definitions for canary suites."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from canary.compare.thresholds import (
    DEFAULT_THRESHOLDS,
    NIGHTLY_THRESHOLDS,
    PERF_THRESHOLDS,
    SMOKE_THRESHOLDS,
    Thresholds,
)


class TestTier(str, Enum):
    """Test tier levels with different time/thoroughness tradeoffs."""

    SMOKE = "smoke"  # 5-10 min, PR gating
    PERF = "perf"  # 20-45 min, performance analysis
    NIGHTLY = "nightly"  # 1-3 hrs, comprehensive soak


class TestDefinition(BaseModel):
    """Definition of a canary test in the matrix."""

    name: str
    description: str
    tier: TestTier
    config_path: str
    thresholds: Thresholds
    enabled: bool = True
    timeout_minutes: int = 60


# Default test matrix
DEFAULT_TEST_MATRIX: list[TestDefinition] = [
    TestDefinition(
        name="dpo_smoke",
        description="DPO smoke test for PR gating",
        tier=TestTier.SMOKE,
        config_path="configs/dpo_smoke.yaml",
        thresholds=SMOKE_THRESHOLDS,
        timeout_minutes=15,
    ),
    TestDefinition(
        name="sft_smoke",
        description="SFT smoke test for PR gating",
        tier=TestTier.SMOKE,
        config_path="configs/sft_smoke.yaml",
        thresholds=SMOKE_THRESHOLDS,
        timeout_minutes=15,
    ),
    TestDefinition(
        name="dpo_perf",
        description="DPO performance test with profiling",
        tier=TestTier.PERF,
        config_path="configs/dpo_perf.yaml",
        thresholds=PERF_THRESHOLDS,
        timeout_minutes=60,
    ),
]


def get_tests_for_tier(tier: TestTier) -> list[TestDefinition]:
    """Get all enabled tests for a specific tier.

    Args:
        tier: Test tier to filter by.

    Returns:
        List of test definitions for the tier.
    """
    return [t for t in DEFAULT_TEST_MATRIX if t.tier == tier and t.enabled]


def get_test_by_name(name: str) -> TestDefinition | None:
    """Get a test definition by name.

    Args:
        name: Test name to look up.

    Returns:
        TestDefinition if found, None otherwise.
    """
    for test in DEFAULT_TEST_MATRIX:
        if test.name == name:
            return test
    return None


def get_thresholds_for_tier(tier: TestTier | str) -> Thresholds:
    """Get thresholds for a test tier.

    Args:
        tier: Test tier (enum or string).

    Returns:
        Appropriate thresholds for the tier.
    """
    if isinstance(tier, str):
        try:
            tier = TestTier(tier)
        except ValueError:
            return DEFAULT_THRESHOLDS

    thresholds_map = {
        TestTier.SMOKE: SMOKE_THRESHOLDS,
        TestTier.PERF: PERF_THRESHOLDS,
        TestTier.NIGHTLY: NIGHTLY_THRESHOLDS,
    }
    return thresholds_map.get(tier, DEFAULT_THRESHOLDS)


class TestMatrix(BaseModel):
    """A collection of tests to run together."""

    name: str
    description: str = ""
    tests: list[TestDefinition]

    @classmethod
    def for_pr(cls) -> "TestMatrix":
        """Create a test matrix for PR gating (smoke tests only)."""
        return cls(
            name="pr_gate",
            description="PR gating tests - quick validation",
            tests=get_tests_for_tier(TestTier.SMOKE),
        )

    @classmethod
    def for_nightly(cls) -> "TestMatrix":
        """Create a test matrix for nightly runs (all tiers)."""
        return cls(
            name="nightly",
            description="Nightly comprehensive tests",
            tests=[t for t in DEFAULT_TEST_MATRIX if t.enabled],
        )

    @classmethod
    def for_performance(cls) -> "TestMatrix":
        """Create a test matrix for performance analysis."""
        return cls(
            name="performance",
            description="Performance analysis tests",
            tests=get_tests_for_tier(TestTier.PERF),
        )
