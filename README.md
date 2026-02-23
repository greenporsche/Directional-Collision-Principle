# Directional Collision Principle

**A Geometric Theory of Adversarial Vulnerability in Neural Networks**

*Siddharth Krishna* | ISEF 2025 | Systems Software / Machine Learning

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

```
├── paper/                          # Research paper and presentation materials
│   ├── FINAL_PAPER_DRAFT.md        # Complete paper draft
│   ├── RESEARCH_SUMMARY.md         # Extended research summary
│   ├── PITCH_SCRIPT.md             # Presentation script
│   ├── POSTER_DESIGN.md            # Poster layout specification
│   └── QA_COMPREHENSIVE.md         # Anticipated Q&A
│
├── research/                       # Theory and background
│   ├── theorem.md                  # Formal theorem statements and proofs
│   ├── application.md              # Theory-to-experiment bridge
│   ├── literature.md               # Literature review (15 papers)
│   ├── methodology.md              # Experimental methodology
│   └── sparse-circuits/            # Reference materials
│       ├── circuit_sparsity-main/  # OpenAI sparse circuits (git submodule)
│       ├── 2511.13653v1.pdf        # Gao et al. 2025 paper
│       └── 2511.13653v1.md         # Paper notes
│
└── experiments/                    # All experimental code and results
    ├── README.md                   # Experiment-specific documentation
    ├── pyproject.toml              # Python dependencies
    ├── src/                        # Experiment scripts
    ├── data/                       # CSV/JSON/TXT result files
    └── figures/                    # Generated PNG and PDF figures
```

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

# Core validation experiments (run from experiments/ directory)
python src/directional_analysis.py       # Directional correlation (r = -0.60)
python src/predict_then_attack.py        # Prediction accuracy (84%)
python src/attack_taxonomy_analysis.py   # Attack strategy mapping (15x efficiency)
python src/lipschitz_estimation.py       # Bracket task L ≈ 3.04
python src/quote_lipschitz_estimation.py # Quote task L ≈ 1.35
python src/quote_bracket_comparison.py   # Cross-task comparison

# Additional analyses
python src/error_analysis.py             # Prediction error analysis
python src/defense_demonstration.py      # Margin-based defense
python src/theorem_validation.py         # Theorem R4 validation
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

## Citation

```bibtex
@misc{krishna2025directional,
  title={Directional Collision Principle: A Geometric Theory of Adversarial Vulnerability in Neural Networks},
  author={Krishna, Siddharth},
  year={2025},
  note={ISEF 2025 Competition Entry}
}
```

---

## License

This code is for educational and research purposes. The sparse circuit models are provided by OpenAI under their [Apache 2.0 license](https://github.com/openai/circuit_sparsity/blob/main/LICENSE).
