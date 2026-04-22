# Directional Collision Principle

**A Geometric Theory of Adversarial Vulnerability in Neural Networks**

*Siddharth Krishna* | Systems Software / Machine Learning

---

## Abstract

This research develops and validates the **Directional Collision Principle**, a mathematical framework that predicts which neural network inputs are vulnerable to adversarial attack. By combining Lipschitz continuity bounds with the symmetry structure of sparse circuits, we derive an input-space distance lower bound and a family-specific directional budget formalism that together explain when and why attacks succeed.

The framework was validated on OpenAI's interpretable sparse circuits across two independent tasks:

| Metric | Value |
|--------|-------|
| Prediction accuracy (identifying vulnerable inputs before attack) | **87.5%** |
| Recall (zero missed vulnerabilities) | **100%** |
| Attack efficiency improvement over random perturbations | **15.5x** [95% CI: 11.2x-19.8x] |
| Directional correlation (margin vs. attack success) | **r = -0.603** [95% CI: -0.78, -0.35] |
| Cross-task normalized margin ratio | **3.56x** |

---

## Repository Structure

- **`paper/`** — Research paper, presentation materials, and Q&A prep
- **`research/`** — Formal theory (theorems, proofs), literature review, and OpenAI sparse circuits submodule
- **`experiments/`** — All experimental code (`src/`), result data (`data/`), and generated figures (`figures/`)

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/greenporsche/Directional-Collision-Principle.git
cd Directional-Collision-Principle

# Set up the experiments environment
cd experiments
uv sync

# Install the sparse circuits package from submodule
uv pip install -e ../research/sparse-circuits/circuit_sparsity-main
```

### Running Experiments

```bash
cd experiments
source .venv/bin/activate

# Run all core experiments in sequence
python src/main.py
```

Each script can also be run individually:

```bash
# Core validation
python src/directional_analysis.py       # Directional correlation (r = -0.60)
python src/predict_then_attack.py        # Prediction accuracy (87.5%)
python src/attack_taxonomy_analysis.py   # Attack strategy mapping (15.5x efficiency)

# Lipschitz estimation
python src/lipschitz_estimation.py       # Bracket task L ≈ 3.04
python src/quote_lipschitz_estimation.py # Quote task L ≈ 1.35
python src/quote_bracket_comparison.py   # Cross-task comparison

# Additional analyses
python src/error_analysis.py             # Prediction error analysis
python src/defense_demonstration.py      # Margin-based defense
```

Figures are saved to `experiments/figures/` and data outputs to `experiments/data/`.

---

## Models

| Model | Parameters | Task | Source |
|-------|-----------|------|--------|
| csp_yolo2 | 475M | Bracket counting (`]]` vs `]`) | [OpenAI](https://github.com/openai/circuit_sparsity) |
| csp_yolo1 | 159M | Quote type (`"` vs `'`) | [OpenAI](https://github.com/openai/circuit_sparsity) |

Models are automatically downloaded from OpenAI's public blob storage on first run.

---

## License

This code is for educational and research purposes. The sparse circuit models are provided by OpenAI under their [Apache 2.0 license](https://github.com/openai/circuit_sparsity/blob/main/LICENSE).
