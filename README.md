# RLHF Canary

Regression detection for RLHF/finetuning pipelines. Automatically catch performance, stability, and correctness regressions before they waste researcher time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/01_quickstart.ipynb)

**No local GPU? No problem.** Try RLHF Canary with a free T4 GPU in Google Colab (~15 min):
- Run a DPO canary training job
- Save metrics as a baseline
- Compare runs and detect regressions

### Tutorials

| Notebook | Description | |
|----------|-------------|---|
| [01_quickstart](notebooks/01_quickstart.ipynb) | Core workflow for detecting regressions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/01_quickstart.ipynb) |
| [02_profiler_deep_dive](notebooks/02_profiler_deep_dive.ipynb) | PyTorch profiler for GPU bottlenecks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/02_profiler_deep_dive.ipynb) |
| [03_stability_monitoring](notebooks/03_stability_monitoring.ipynb) | NaN/loss divergence detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/03_stability_monitoring.ipynb) |
| [04_root_cause_analysis](notebooks/04_root_cause_analysis.ipynb) | Heuristics for debugging failures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/04_root_cause_analysis.ipynb) |
| [05_ppo_canary](notebooks/05_ppo_canary.ipynb) | PPO-specific training validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/05_ppo_canary.ipynb) |
| [06_sft_canary](notebooks/06_sft_canary.ipynb) | SFT training validation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/06_sft_canary.ipynb) |
| [07_ci_cd_integration](notebooks/07_ci_cd_integration.ipynb) | GitHub Actions & PR gating | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/07_ci_cd_integration.ipynb) |
| [08_configuration_and_thresholds](notebooks/08_configuration_and_thresholds.ipynb) | Custom thresholds & test matrices | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/08_configuration_and_thresholds.ipynb) |
| [09_quantization_and_memory](notebooks/09_quantization_and_memory.ipynb) | 4-bit/8-bit quantization & baselines | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/09_quantization_and_memory.ipynb) |
| [vscode_dev](notebooks/vscode_dev.ipynb) | Local VS Code + Colab development | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmcmanus1/rlhf-canary/blob/main/notebooks/vscode_dev.ipynb) |

## Features

- **Performance regression detection**: Track tokens/sec, step time, GPU utilization, memory usage
- **Stability monitoring**: Detect NaN/Inf values, loss divergence, gradient explosion
- **Root cause analysis**: Heuristic-based diagnosis of regression causes
- **CI/CD integration**: GitHub Actions workflow with PR gating
- **Flexible configuration**: YAML-based configs for smoke, perf, and nightly tests

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rlhf-canary.git
cd rlhf-canary

# Install with pip
pip install -e ".[dev]"

# Or install dependencies directly
pip install -r requirements.txt
```

### Run a canary test

```bash
# Show environment info
canary env

# Run DPO smoke test
canary run configs/dpo_smoke.yaml

# The output will be saved to ./canary_output/<run_id>/metrics.json
```

### Compare to baseline

```bash
# Save current run as baseline
canary save-baseline ./canary_output/<run_id>/metrics.json ./baselines/dpo_smoke.json

# Run again and compare
canary run configs/dpo_smoke.yaml
canary compare ./canary_output/<new_run_id>/metrics.json ./baselines/dpo_smoke.json
```

## CLI Commands

```bash
canary --help                    # Show all commands
canary env                       # Show environment fingerprint
canary run <config>              # Run a canary job
canary compare <current> <base>  # Compare metrics to baseline
canary save-baseline <src> <dst> # Save metrics as baseline
canary init-config <path>        # Generate sample config
```

## Configuration

Canary jobs are configured via YAML files:

```yaml
# configs/dpo_smoke.yaml
name: dpo_smoke
description: DPO smoke test

# Model
model_name: EleutherAI/pythia-70m
use_peft: true
lora_r: 16

# Training
training_type: dpo
max_steps: 100
batch_size: 2
gradient_accumulation_steps: 4

# DPO-specific
beta: 0.1

# Dataset
dataset_name: Anthropic/hh-rlhf
dataset_size: 512
```

### Test Tiers

| Tier | Steps | Duration | Use Case |
|------|-------|----------|----------|
| smoke | 100 | ~5 min | PR gating |
| perf | 500 | ~20 min | Detailed perf analysis |
| nightly | 2000 | ~2 hr | Comprehensive soak |

## Regression Detection

### What's checked

| Category | Metric | Default Threshold |
|----------|--------|-------------------|
| Performance | Step time increase | 10% |
| Performance | Tokens/sec drop | 8% |
| Performance | Memory increase | 500MB or 20% |
| Stability | NaN steps | 0 allowed |
| Stability | Inf steps | 0 allowed |
| Stability | Loss divergence | Auto-detected |

### Example output

```
# RLHF Canary Report ✅

## Summary
**Status:** PASS
**Baseline Run:** `dpo_smoke_1234_abcd`
**Current Run:** `dpo_smoke_5678_efgh`

## Regression Checks
| Check | Status | Baseline | Current | Change | Threshold |
|-------|--------|----------|---------|--------|-----------|
| nan_steps | ✅ | 0 | 0 | - | 0 |
| step_time_mean | ✅ | 0.4523s | 0.4601s | +1.7% | 10% |
| tokens_per_sec | ✅ | 1847 | 1820 | -1.5% | 8% |
| max_memory | ✅ | 2341MB | 2356MB | +0.6% | 500MB |
```

## Root Cause Analysis

When regressions are detected, the canary provides heuristic analysis:

```markdown
## Root Cause Analysis

**Summary:** Most likely cause: dataloader (Dataloader or preprocessing bottleneck)

### #1 Dataloader (████████░░ 70%)
Dataloader or preprocessing bottleneck

**Evidence:**
- Step time increased by 25.0%
- Large increases often indicate CPU-side bottlenecks

**Suggested Actions:**
- Check dataloader num_workers configuration
- Profile CPU utilization during training
- Check for tokenization changes
```

## GitHub Actions Integration

Add the workflow to your repo:

```yaml
# .github/workflows/canary.yml
name: RLHF Canary

on:
  pull_request:
    branches: [main]

jobs:
  canary:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .
      - run: canary run configs/dpo_smoke.yaml
      - run: canary compare ./canary_output/*/metrics.json ./baselines/baseline.json
```

## Project Structure

```
rlhf-canary/
├── canary/
│   ├── cli.py              # CLI interface
│   ├── runner/             # Job execution
│   │   ├── base.py         # Base runner
│   │   └── local.py        # Local runner
│   ├── collect/            # Metrics collection
│   │   ├── metrics.py      # Training callbacks
│   │   ├── profiler.py     # PyTorch profiler
│   │   └── env_fingerprint.py
│   ├── compare/            # Regression detection
│   │   ├── stats.py        # Statistical comparison
│   │   ├── thresholds.py   # Configurable thresholds
│   │   └── heuristics.py   # Root cause analysis
│   └── report/             # Output generation
│       ├── markdown.py     # Markdown reports
│       └── github.py       # GitHub integration
├── configs/                # Sample configurations
├── tests/                  # Unit tests
└── workflows/              # CI/CD workflows
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=canary --cov-report=html

# Lint
ruff check canary/
```

## Use Cases

### PR Regression Gate
Run smoke tests on every PR to catch obvious regressions:
```bash
canary run configs/dpo_smoke.yaml
canary compare ./current/metrics.json ./baselines/main.json --threshold-tier smoke
```

### Nightly Soak Test
Run longer tests overnight to catch "slowdown after N steps":
```bash
canary run configs/dpo_perf.yaml
canary compare ./current/metrics.json ./baselines/main.json --threshold-tier nightly
```

### New Architecture Validation
When changing model architecture, validate training stability:
```bash
canary run configs/dpo_perf.yaml
# Check for NaN, loss divergence, memory changes
```

## License

MIT
