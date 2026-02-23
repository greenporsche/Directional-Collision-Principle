# Adversarial Robustness Experiments

## Directional Collision Principle: Predicting Neural Network Vulnerability

This directory contains all experimental code for validating the Directional Collision Principle on OpenAI's sparse circuit models.

---

## Quick Start

### Prerequisites

- Python 3.11+
- uv (recommended) or pip

### Installation

```bash
cd experiments

# Create virtual environment and install dependencies
uv sync

# Install the sparse circuits package (git submodule)
uv pip install -e ../research/sparse-circuits/circuit_sparsity-main
```

### Running Experiments

All scripts should be run from the `experiments/` directory.

```bash
source .venv/bin/activate

# Core validation experiments
python src/directional_analysis.py       # Core finding: directional correlation
python src/predict_then_attack.py        # Prediction accuracy
python src/attack_taxonomy_analysis.py   # Attack strategy mapping
python src/lipschitz_estimation.py       # Estimate L for bracket task
python src/quote_lipschitz_estimation.py # Estimate L for quote task
python src/quote_task_validation.py      # Cross-task validation (quote task)
python src/quote_bracket_comparison.py   # Compare tasks with task-specific L

# Error analysis and defense
python src/error_analysis.py             # Analyze prediction errors
python src/defense_demonstration.py      # Show defense effectiveness
```

---

## Project Structure

```
experiments/
├── src/                               # Experiment scripts
│   ├── directional_analysis.py        # KEY FINDING: Directional correlation
│   ├── predict_then_attack.py         # 84% prediction accuracy
│   ├── attack_taxonomy_analysis.py    # Attack strategy mapping
│   ├── lipschitz_estimation.py        # Bracket task L ≈ 3.04
│   ├── quote_lipschitz_estimation.py  # Quote task L ≈ 1.35
│   ├── quote_task_validation.py       # Cross-task validation
│   ├── quote_bracket_comparison.py    # Bracket vs quote comparison
│   ├── error_analysis.py             # Prediction error analysis
│   ├── defense_demonstration.py       # Margin-based defense
│   ├── experiment_framework.py        # Base classes and utilities
│   ├── circuit_informed_attack.py     # Attack implementations
│   ├── gradient_attack.py            # Gradient-based attacks
│   ├── hero_figure.py                # Poster centerpiece figure
│   ├── visualize_results.py          # Additional visualizations
│   ├── theorem_validation.py         # Theorem R4 validation
│   ├── symmetry_visualization.py     # Symmetry structure figures
│   ├── lipschitz_inflation_analysis.py # L-inflation analysis
│   ├── quote_task_theoretical.py     # Theoretical predictions
│   ├── certified_lipschitz_bound.py  # Certified bounds analysis
│   ├── icp_validation.py            # Independent calibration probe
│   ├── run_full_experiment.py        # Full experiment runner
│   ├── multi_seed_baseline.py        # Multi-seed baseline
│   └── main.py                       # Entry point
│
├── data/                              # Results and data files
│   ├── baseline_results.csv           # Clean model predictions (32 samples)
│   ├── full_experiment_results.csv    # All attack trials
│   ├── circuit_informed_results.csv   # Circuit-targeted attacks
│   ├── quote_task_results.csv         # Quote task validation
│   ├── estimated_parameters.json      # L, K estimates
│   └── *_results.txt                  # Per-experiment summaries
│
├── figures/                           # Generated figures
│   ├── hero_figure_complete.png/pdf   # Poster centerpiece
│   ├── directional_analysis.png       # Core validation
│   ├── predict_then_attack.png        # Prediction accuracy
│   ├── attack_taxonomy.png/pdf        # Strategy comparison
│   ├── lipschitz_estimation.png/pdf   # Bracket task L
│   ├── quote_lipschitz_estimation.png/pdf # Quote task L
│   ├── quote_bracket_comparison.png/pdf   # Cross-task
│   ├── error_analysis.png/pdf         # Error breakdown
│   ├── defense_demonstration.png/pdf  # Defense effectiveness
│   └── ...                            # Additional figures
│
├── pyproject.toml                     # Dependencies
├── .python-version                    # Python 3.12
├── uv.lock                           # Locked dependencies
├── .gitignore
└── README.md                          # This file
```

---

## Key Results

| Metric | Value | Script |
|--------|-------|--------|
| Directional Correlation | r = -0.60 (p < 0.001) | `src/directional_analysis.py` |
| Prediction Accuracy | 84.4% | `src/predict_then_attack.py` |
| Recall (no missed attacks) | 100% | `src/predict_then_attack.py` |
| Attack Efficiency Improvement | 15.5x | `src/attack_taxonomy_analysis.py` |
| Bracket Task L | **3.04** | `src/lipschitz_estimation.py` |
| Quote Task L | **1.35** | `src/quote_lipschitz_estimation.py` |
| L Ratio (sensitivity) | **2.24x** | `src/quote_bracket_comparison.py` |
| Normalized Margin Ratio | **3.56x** | `src/quote_bracket_comparison.py` |

---

## Reproducing Results

### Step 1: Verify baseline predictions

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/baseline_results.csv')
print(f'Samples: {len(df)}')
print(f'Model accuracy: {(df[\"correct\"].mean())*100:.1f}%')
"
```

Expected: 32 samples, ~94% baseline accuracy

### Step 2: Run directional analysis

```bash
python src/directional_analysis.py
```

Expected output:
- Random attacks: r ~ 0 (no correlation)
- Circuit attacks: r ~ -0.60 (strong negative correlation)

### Step 3: Run prediction protocol

```bash
python src/predict_then_attack.py
```

Expected output:
- Accuracy: ~84%
- Recall: 100%
- ROC AUC: ~0.82

### Step 4: Run cross-task validation

```bash
python src/quote_lipschitz_estimation.py  # Estimate L for quote task
python src/quote_task_validation.py
python src/quote_bracket_comparison.py
```

Expected:
- Quote task L ~ 1.35 (vs bracket L ~ 3.04)
- L ratio: 2.24x (bracket is more sensitive)
- Normalized margin ratio: 3.56x

---

## Dependencies

Core dependencies (see `pyproject.toml`):
- `pandas>=2.0.0`
- `numpy>=1.24.0,<2`
- `scipy>=1.11.0`
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `scikit-learn>=1.3.0`
- `torch>=2.0.0`
- `tiktoken>=0.5.0`

Additional: `circuit-sparsity` (OpenAI's package, installed from git submodule)

---

## Model Information

| Model | Parameters | Task | Source |
|-------|-----------|------|--------|
| csp_yolo2 | 475M | Bracket counting (`]]` vs `]`) | OpenAI |
| csp_yolo1 | 159M | Quote type (`"` vs `'`) | OpenAI |

Models are automatically downloaded from OpenAI's public blob storage on first run.

---

## Data Format

### circuit_informed_results.csv

| Column | Description |
|--------|-------------|
| sample_id | Unique sample identifier (0-31) |
| attack_type | Attack category (circuit_informed) |
| attack_strategy | Specific strategy (distractor_comment, etc.) |
| tokens_modified | Perturbation budget (1, 2, 3, 5) |
| original_prediction | Model's prediction on clean input |
| attacked_prediction | Model's prediction after attack |
| prediction_flipped | Whether attack succeeded (bool) |
| original_logit_diff | logit("]]") - logit("]") before attack |
| attacked_logit_diff | logit("]]") - logit("]") after attack |

---

## Troubleshooting

### NumPy version error
```
Error: A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Fix:** Ensure numpy < 2 is installed (already constrained in pyproject.toml):
```bash
uv pip install "numpy>=1.24.0,<2"
```

### Model download fails
```
HTTPError: 404 Not Found
```

**Fix:** Check internet connection and verify OpenAI blob storage is accessible.

### CUDA not available
```
RuntimeError: CUDA not available
```

**Fix:** The code defaults to CPU. For GPU support, install CUDA-enabled PyTorch.
