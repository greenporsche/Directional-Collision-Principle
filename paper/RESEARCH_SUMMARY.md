# Directional Collision Principle: A Geometric Theory of Adversarial Vulnerability in Neural Networks

## Research Summary

**Author:** Siddharth Krishna
**Category:** Systems Software / Computational Biology and Bioinformatics / Mathematics

---

## Abstract

This research develops and validates the **Directional Collision Principle**, a novel mathematical framework that identifies the geometric conditions for adversarial vulnerability in neural networks. By combining Lipschitz continuity bounds from certified robustness theory with the structural symmetries of sparse circuits, I derived a proved distance lower bound (Theorem R4) showing that adversarial collisions require a minimum perturbation distance determined by the model's margin and Lipschitz scale.

The theory was validated on OpenAI's weight-sparse transformers across two independent tasks. Key findings include:
- **15.5x attack efficiency improvement** [95% CI: 11.2x, 19.8x] over random perturbations by exploiting circuit structure.
- **87.5% prediction accuracy** in identifying vulnerable inputs using a calibrated margin-based decision rule.
- **r = -0.60 correlation** [95% CI: -0.78, -0.35] between directional margin and attack success on the bracket task.
- **Functional Asymmetry Discovery:** The quote task's sign-reversal ($r = +0.37$) was found to be driven by model-attack interaction asymmetry (13.4% one-way fragility), validating the need for directional family constraints in robustness analysis.

I estimated task-specific sensitivity parameters ($\hat{L}_{bracket} \approx 3.04, \hat{L}_{quote} \approx 1.35$), enabling cross-task generalization via normalized margins. This work demonstrates that while mechanistic interpretability exposes model vulnerabilities, it simultaneously provides the rigorous foundation necessary for certified defense heuristics.

---

## 1. Research Question

> **How accurately can a theory grounded in Lipschitz geometry and directional symmetry predict the vulnerability of sparse neural circuit representations, and how do directional constraints influence realized attack efficiency?**

### Gradient Framework Classification
- **Level:** 8M (Resource-Bounded Optimization with Mechanistic Probing)
- **Constraints satisfied:** C1-C11 (Coherence through Mechanism)

---

## 2. Background and Motivation

### 2.1 The Problem of Adversarial Vulnerability

Neural networks are susceptible to adversarial attacks—small, carefully crafted perturbations that cause misclassification. Despite extensive empirical research, a comprehensive theoretical framework explaining *when* and *why* attacks succeed has been lacking.

### 2.2 Interpretable Sparse Circuits

Recent work by OpenAI (Gao et al., 2025) introduced **weight-sparse transformers** that learn disentangled, interpretable circuits. These models:
- Enforce L₀ sparsity during training (most weights are exactly zero)
- Learn minimal circuits for specific tasks (16× smaller than dense models)
- Enable mechanistic understanding of model behavior

This interpretability creates both an opportunity and a risk: understanding circuits enables better attacks, but also better defenses.

### 2.3 Research Gap

Prior work focused on input-space perturbation bounds (ε-ball robustness), empirical attack/defense methods, and post-hoc interpretability. Missing is a **representation-space theory** connecting interpretability to vulnerability, with empirically validated Lipschitz bounds and directional sensitivity analysis.

---

## 3. Theoretical Framework

### 3.1 Epistemic Labeling and Scope
To avoid theorem-experiment conflation, we classify all statements as **PROVED**, **CALIBRATED**, or **OBSERVED**.

- **PROVED**: mathematical statements that follow from explicitly stated assumptions.
- **CALIBRATED**: practical instantiations where unknown theoretical constants are replaced by empirical surrogates.
- **OBSERVED**: empirical regularities in finite experiments.

This labeling is used throughout Sections 3 to 7.

### 3.2 Assumptions
We work under assumptions A1-A9 (listed in Section 3.8). The essential structural distinction is:
1. The **output-space involution geometry** is exact and requires only A7.
2. The **input-space robustness inequality** requires A1-A6.
3. **Directional attack asymmetry** requires A8 and (for local sparse tightening) optionally A9.

### 3.3 Output-Space Involution Geometry (**PROVED**)
Let $V_{out} = \mathbb{R}^2$ be the two-logit space with $\tau_{out}(z_1,z_2) = (z_2,z_1)$. Define $g_{out}(z) = z_1 - z_2$. Then $g_{out}(\tau_{out}(z)) = -g_{out}(z)$, and the zero set is $Z_{out} = \{z : z_1 = z_2\}$. This gives an exact odd-function geometry in logit space.

Interpretation: output-space symmetry is mathematically clean but does not by itself certify robustness in input space.

### 3.4 Input-Space Robustness Bound (Theorem R4, **PROVED**)
Define margin $m(x) = f_1(x) - f_2(x)$, and boundary set $D = \{x : m(x)=0\}$. Under A1-A5, $m$ is $(G_1+G_2)$-Lipschitz because
$|m(x)-m(y)| \leq |f_1(x)-f_1(y)| + |f_2(x)-f_2(y)| \leq (G_1+G_2)d_X(x,y)$.
Therefore, for every $x$:

$$dist_X(x,D) \geq \frac{|m(x)|}{G_1+G_2}$$

This is the core theorem-level statement used by the paper: a necessary minimum movement to reach the decision boundary.

### 3.5 Empirical Instantiation with $\hat{L}$ (**CALIBRATED**)
In practice, $(G_1+G_2)$ is unknown for transformer models. We therefore instantiate the bound with an empirically calibrated sensitivity parameter $\hat{L}$ (95th percentile slope estimate under the declared attack/metric protocol):

$$\text{score}(x) = \frac{|m(x)|}{\hat{L}}$$

This score is operationally useful for ranking vulnerability and predicting relative attack budget, but it is **not** a certified radius unless A6 holds with a provable global upper bound.

### 3.6 Directional Family Constraint (**PROVED** structure, **OBSERVED** parameters)
Let $F$ denote a fixed attack family with admissible edit set $\Delta_F(x)$. Define per-edit family gains:
- $\beta^+(F)$: maximal one-edit increase in $m$,
- $\beta^-(F)$: maximal one-edit decrease in $m$.

For any $t$-edit sequence from family $F$:
- $m_{after} \leq m_{before} + t \cdot \beta^+(F)$
- $m_{after} \geq m_{before} - t \cdot \beta^-(F)$

Hence crossing requirements are:
- if $m_{before} < 0$, reaching boundary requires $t \geq |m_{before}|/\beta^+(F)$
- if $m_{before} > 0$, reaching boundary requires $t \geq |m_{before}|/\beta^-(F)$

This explains why equal absolute margins can yield unequal success rates by direction: attack success depends on both margin magnitude and family-dependent directional budget.

### 3.7 Sparse-Circuit Local Tightening (**PROVED locally under A9**)
Let $c_i(x)$ bound the one-token influence on $m$ at coordinate $i$ near $x$. If only coordinates in active set $S_x$ materially influence $m$ locally (A9), then for substitution perturbations:

$$|\Delta m| \leq \sum_{i \in \text{changed} \cap S_x} c_i(x)$$

Consequences:
1. Effective local sensitivity shrinks when perturbations mostly avoid $S_x$.
2. Structured attacks that overlap $S_x$ can be much more efficient than random edits.
3. Sparse interpretability (identifying $S_x$) can tighten practical, family-specific risk estimates.

### 3.8 Named Assumptions Table
| ID | Name | Formal Statement | Status |
|----|------|------------------|--------|
| A1 | Input space | $X$ is a set of token sequences | Standard |
| A2 | Metric | $d_X$ is Hamming distance (substitutions) | Declared |
| A3 | Output Map | $F(x) = (f_1(x), f_2(x))$ logits | Observed |
| A4 | Boundary | $D = \{x : f_1(x) = f_2(x)\}$ | Definition |
| A5 | Regularity | $f_i$ are Lipschitz constants $G_i$ | Assumption |
| A6 | Certification | True $G \geq G_1+G_2$ exists | Unknown |
| A7 | Involution | $\tau_{out}$ swaps logits | Proved |
| A8 | Admissibility | Attack family defines $\Delta_F(x)$ | Declared |
| A9 | Local Sparsity | $m$ locally depends only on $S_x$ tokens | Testable |

### 3.9 What Section 3 Proves and What It Does Not
- **Proved**: lower bound on required distance to boundary (Theorem R4); family-directional budget inequalities; local sparse tightening under A9.
- **Calibrated**: replacing unknown global constant by $\hat{L}$ for operational prediction.
- **Observed**: quote-task directional asymmetry, bracket-task strong negative margin-success correlation, and finite-sample predictive performance.

---

## 4. Experimental Methodology

### 4.1 Input Metric and V-Space Definition
We define the input space $V$ as the set of discrete token sequences of fixed length $N$. To enable Lipschitz analysis, we equip this space with the **Hamming Distance** $d_H(v, v') = \sum_{i=1}^N \mathbb{1}[v_i \neq v'_i]$, which counts the number of token substitutions required to transform one sequence into another. This metric aligns with our experimental perturbation budget (tokens modified).

### 4.2 Sensitivity Calibration Protocol (**CALIBRATED**)
The parameter $\hat{L}$ is estimated empirically to instantiate the theoretical distance bound (Section 3.4). For each task, we compute the empirical sensitivity $s = |\Delta g(v)| / d_H(v, v')$ across all attack trials. We define $\hat{L}$ as the **95th percentile** of these observed slopes. This provides a robust, task-specific calibration that accounts for the high variance in token-level influence while filtering extreme outliers.

### 4.3 Model Systems and Tasks
We validate the framework on OpenAI’s weight-sparse transformers (Gao et al., 2025):
1. **Bracket Counting (csp_yolo2, 475M params):** Binary classification of "]]" vs "]" closing tokens.
2. **Quote Detection (csp_yolo1, 159M params):** Binary classification of '"' vs "'" closing tokens.

### 4.4 Attack Strategy Taxonomy
We implement and compare three categories of circuit-informed attacks:
- **Radius-Exceeding (Unbalanced Context):** Pushes the input beyond the exclusion radius by inserting unmatched structural tokens.
- **Involution-Targeted (Distractor/Activation):** Exploits mechanistic knowledge of the task circuit to apply perturbations in high-influence directions.
- **Lipschitz Inflation (Context Dilution):** Increases the effective local sensitivity by saturating the context with distractor tokens.

### 4.5 Independent Calibration Probe (ICP) Protocol
To resolve circularity in local sensitivity estimation, we introduce the **ICP protocol**. Before executing a targeted attack, an input $v$ is subjected to $M=10$ random single-token substitutions sampled from the `tinypython` vocabulary distribution. The mean logit shift across these probes defines the **Prospective Local Lipschitz Estimate** ($\hat{L}_{local}$), allowing for vulnerability prediction prior to the actual adversarial attempt.

---

## 5. Experimental Results

### 5.1 Directional Validation and Attack Efficiency (**OBSERVED**)
Circuit-informed attacks achieved a success rate of **24.2% [95% CI: 18.1%, 30.3%]** across all trials, compared to a baseline success rate of **1.56% [95% CI: 0.2%, 3.8%]** for random perturbations. This constitutes a **15.5x efficiency improvement [95% CI: 11.2x, 19.8x]** ($p < 0.0001$, Fisher's exact test).

On the bracket task, we observed a strong negative correlation between signed margin and attack success ($r = -0.603$ [95% CI: -0.78, -0.35], $p = 0.0003$). This validates the core prediction of the **Directional Collision Principle**: vulnerability is a function of both boundary distance and directional alignment.

### 5.2 Calibrated Prediction Accuracy (**CALIBRATED**)
Using the binary decision rule (predict VULNERABLE if logit_diff < 0), we achieved **87.5% prediction accuracy** with **100% recall** (0 missed vulnerabilities) on the bracket task. When instantiated with the calibrated parameter $\hat{L} = 3.04$, the resulting adversarial margin correctly identifies the required token budget for successful flips in 28 out of 32 samples.

### 5.3 Functional Asymmetry Discovery (**OBSERVED**)
Validation on the quote task revealed a positive correlation ($r = +0.368, p = 0.037$). This sign reversal was found to be driven by a profound **Functional Asymmetry** in the model’s response surface:
- Attacks pushing `"` $\rightarrow$ `'` achieved **13.4%** success.
- Attacks pushing `'` $\rightarrow$ `"` achieved **0.0%** success.

This asymmetry indicates that the double-quote representation is mechanistically more fragile than the single-quote representation in the `csp_yolo1` model. This finding demonstrates that while the theorem (Section 3.4) provides the minimum distance bound, the realized vulnerability is constrained by family-specific **Directional Budgets** (Section 3.6).

### 5.4 Cross-Task Sensitivity and Normalized Margins (**CALIBRATED**)
We identified task-specific sensitivity parameters:
- **Bracket Task:** $\hat{L} = 3.04$
- **Quote Task:** $\hat{L} = 1.35$

The bracket counting circuit is **2.24x more sensitive** to perturbations than the quote detection circuit. By computing **Normalized Margins** ($|m(x)| / \hat{L}(1+K)$), we correctly predicted the 3.5x larger attack budget required for the quote task, confirming the framework's cross-task generalizability.

### 5.5 Root Cause Analysis: Local Lipschitz Variation (**OBSERVED**)
Post-hoc analysis of prediction failures (False Positives) identified a **15x variation in local sensitivity**. FP samples exhibited a mean $\hat{L}_{local} \approx 0.10$, whereas True Positives exhibited $\hat{L}_{local} \approx 1.49$ ($p=0.009$). Preliminary validation of the **ICP protocol** demonstrates that this sensitivity gap can be detected pre-attack using random probes, providing a path toward 100% precision in vulnerability identification.

---

## 6. Discussion

### 6.1 Main Contribution with Correct Claim Levels
This work contributes a two-part theory-to-data framework.

1. **Theorem-level contribution (PROVED):** Theorem R4 gives a necessary lower bound on input-space perturbation distance to decision tie under declared metric and Lipschitz assumptions.
2. **Directional refinement (PROVED structure + OBSERVED parameters):** family-specific directional budget terms ($\beta^+, \beta^-$) explain asymmetric attack outcomes that margin-only analysis cannot.
3. **Operational instantiation (CALIBRATED):** $\hat{L}$ yields a practical margin-based risk score that predicts vulnerability patterns in experiments.

### 6.2 Interpretation of Cross-Task Results
The quote-task positive signed correlation does not falsify Theorem R4. Instead, it indicates asymmetric family-direction budgets in that model-task pair: one direction is substantially easier for the tested perturbation family. Thus, the theorem supplies a necessary distance condition, while directional accessibility determines realized success patterns.

### 6.3 Limits and Non-Claims
- We do **not** claim certified robustness from $\hat{L}$.
- We do **not** claim the directional rule is a universal theorem for all attack families.
- We treat $L_{local}$ findings as post-hoc mechanism evidence pending full prospective ICP validation.
- Margin-based filtering is not validated against adaptive white-box attackers and is therefore a scoped heuristic.

### 6.4 Why This Matters for Interpretable Sparse Circuits
Sparse circuits expose structure (active coordinates, family-specific vulnerabilities) that improves mechanistic understanding of failures. This can increase both attack efficiency (when structure is exploited) and diagnostic power (when structure is monitored). The key scientific point is not that sparsity is uniformly safer; it is that sparsity makes the vulnerability geometry measurable.

---

## 7. Conclusions

This project now supports a precise final claim set:

1. **PROVED:** Under explicit assumptions, distance to decision boundary is lower-bounded by margin over Lipschitz scale (Theorem R4).
2. **PROVED/OBSERVED bridge:** Directional family budgets ($\beta^+, \beta^-$) provide a formal mechanism for asymmetric attack success and explain sign reversals without contradicting the distance theorem.
3. **CALIBRATED:** Empirical $\hat{L}$ instantiation provides practical vulnerability scoring and strong predictive utility on tested sparse-circuit tasks.
4. **OBSERVED:** Quote and bracket tasks exhibit different directional fragilities, indicating model-attack interaction structure beyond margin magnitude alone.

The paper’s central contribution is therefore a disciplined theory-empiric bridge: a proved boundary-distance theorem, an explicitly labeled calibrated instantiation, and empirically validated directional structure.

---

## 8. Key Figures

### Figure 1: Hero Figure (Poster Centerpiece)
`experiments/hero_figure_complete.png`
Four-panel summary: A: DCP diagram; B: Correlation validation; C: Prediction accuracy; D: Key metrics.

### Figure 5: Lipschitz Estimation
`experiments/lipschitz_estimation.png`
A: Distribution of estimates; B: Logit change vs perturbation; C: $\hat{L}$ by strategy; D: Margin formula.

### Figure 8: Error Analysis (Root Cause Discovery)
`experiments/error_analysis.png`
A: Local Lipschitz comparison (15x gap); B: Perfect separation with effective margin; C: Logit shift achieved.

---

## 9. Statistical Summary

| Metric | Value | CI | Claim Type |
|--------|-------|----|------------|
| Bracket correlation | r = -0.603 | [-0.78, -0.35] | OBSERVED |
| Quote correlation | r = +0.368 | p = 0.037 | OBSERVED |
| Prediction accuracy | 87.5% | 100% recall | CALIBRATED |
| Efficiency ratio | 15.5x | [11.2x, 19.8x] | OBSERVED |
| $\hat{L}$ (bracket) | 3.04 | 95th pctile | CALIBRATED |
| $\hat{L}$ (quote) | 1.35 | 95th pctile | CALIBRATED |
| $L_{local}$ ratio (TP/FP) | 15x | p = 0.009 | OBSERVED |
| Normalized margin ratio | 3.56x | cross-task | CALIBRATED |

---

## 10. References

1. Gao, L., et al. (2025). Weight-sparse transformers have interpretable circuits. *arXiv:2511.13653*
2. Szegedy, C., et al. (2014). Intriguing properties of neural networks. *ICLR*.
3. Carlini, N. & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *IEEE S&P*.

---

## 11. Acknowledgments
OpenAI for the sparse circuits codebase.
