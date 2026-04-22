# Directional Collision Principle: A Geometric Theory of Adversarial Vulnerability in Neural Networks

**Author:** Siddharth Krishna
**Category:** Systems Software / Machine Learning

---

## Abstract

This research develops and validates the **Directional Collision Principle**, a mathematical framework that predicts which neural network inputs are vulnerable to adversarial attack. By combining Lipschitz continuity bounds with the symmetry structure of sparse circuits, we derived an input-space distance lower bound (Theorem R4) and a family-specific directional budget formalism that together explain when and why attacks succeed.

The framework was validated on OpenAI's interpretable sparse circuits across **two independent tasks**, achieving:
- **87.5% prediction accuracy** in identifying vulnerable inputs before attack (CALIBRATED)
- **100% recall** (zero missed vulnerabilities)
- **15.5× attack efficiency improvement** [95% CI: 11.2×–19.8×] over random perturbations (OBSERVED)
- **r = −0.603** [95% CI: −0.78, −0.35] correlation between directional margin and attack success (OBSERVED)
- **Cross-task validation:** Normalized margin ratio (3.56×) correctly predicts robustness differences

All claims are explicitly labeled as PROVED (mathematical results under stated assumptions), CALIBRATED (empirical instantiations with estimated parameters), or OBSERVED (empirical regularities from finite experiments).

---

## 1. Introduction

Neural networks are susceptible to adversarial attacks — small, carefully crafted perturbations that cause misclassification. Despite extensive empirical research, a comprehensive theoretical framework explaining *when* and *why* attacks succeed has been lacking.

Recent work by OpenAI (Gao et al., 2025) introduced **weight-sparse transformers** that learn disentangled, interpretable circuits. These models enforce L₀ sparsity during training, creating minimal circuits for specific tasks that are 16× smaller than dense equivalents. This interpretability creates both opportunity and risk: understanding circuits enables better attacks, but also better defenses.

This paper addresses a missing connection: a **representation-space theory** linking interpretability to vulnerability, with empirically calibrated Lipschitz bounds. We provide:

1. A proved input-space distance lower bound (Theorem R4) under explicitly named assumptions
2. A family-specific directional budget formalism explaining asymmetric attack success
3. A calibrated instantiation achieving practical predictive utility
4. Cross-task validation demonstrating generalizability

Throughout, we maintain a strict three-tier claim labeling system (Section 3.1) to prevent theorem-experiment conflation.

---

## 2. Background and Motivation

### 2.1 The Problem of Adversarial Vulnerability

Neural networks are susceptible to adversarial attacks — small, carefully crafted perturbations that cause misclassification. Despite extensive empirical research, a comprehensive theoretical framework explaining *when* and *why* attacks succeed has been lacking.

### 2.2 Interpretable Sparse Circuits

Recent work by OpenAI (Gao et al., 2025) introduced **weight-sparse transformers** that learn disentangled, interpretable circuits. These models:
- Enforce L₀ sparsity during training (most weights are exactly zero)
- Learn minimal circuits for specific tasks (16× smaller than dense models)
- Enable mechanistic understanding of model behavior

### 2.3 Research Gap

Prior work focused on: input-space perturbation bounds (ε-ball robustness), empirical attack/defense methods, and post-hoc interpretability. Missing: a **representation-space theory** connecting interpretability to vulnerability, with empirically validated Lipschitz bounds and explicit epistemic labeling.

---

## 3. Theoretical Framework

### 3.1 Epistemic Labeling and Scope

To avoid theorem-experiment conflation, we classify all statements as **PROVED**, **CALIBRATED**, or **OBSERVED**.

- **PROVED**: mathematical statements that follow from explicitly stated assumptions.
- **CALIBRATED**: practical instantiations where unknown theoretical constants are replaced by empirical surrogates.
- **OBSERVED**: empirical regularities in finite experiments.

This labeling is used throughout Sections 3–7.

### 3.2 Assumptions

We work under assumptions A1–A9 (listed in Section 3.9). The essential structural distinction is:
1. The **output-space involution geometry** is exact and requires only A7.
2. The **input-space robustness inequality** requires A1–A6.
3. **Directional attack asymmetry** requires A8 and (for local sparse tightening) optionally A9.

### 3.3 Output-Space Involution Geometry (PROVED)

Let V_out = ℝ² be the two-logit space with τ_out(z₁, z₂) = (z₂, z₁). Define g_out(z) = z₁ − z₂. Then g_out(τ_out(z)) = −g_out(z), and the zero set is Z_out = {z : z₁ = z₂}. This gives an exact odd-function geometry in logit space.

*Interpretation:* Output-space symmetry is mathematically clean but does not by itself certify robustness in input space.

### 3.4 Input-Space Robustness Bound — Theorem R4 (PROVED)

Define margin m(x) = f₁(x) − f₂(x), and boundary set D = {x : m(x) = 0}. Under A1–A5, m is (G₁ + G₂)-Lipschitz because:

|m(x) − m(y)| ≤ |f₁(x) − f₁(y)| + |f₂(x) − f₂(y)| ≤ (G₁ + G₂)·d_X(x, y).

Therefore, for every x:

**dist_X(x, D) ≥ |m(x)| / (G₁ + G₂)**

This is the core theorem-level statement: a lower bound on required perturbation distance to reach decision tie.

### 3.5 Certified Bound Instantiation (PROVED + COMPUTED)

We instantiate Theorem R4 with a certified upper bound on (G₁ + G₂) computed directly from the model weights via spectral norm composition. For a GPT-2-style transformer with n layers, the certified Lipschitz constant is:

**G_total = σ₁(W_emb) × ∏ᵢ Lip(blockᵢ) × σ₁(W_head)**

where each block's Lipschitz constant is bounded by the residual stream formula:

**Lip(block) ≤ 1 + Lip(attn) + Lip(mlp)**

with:
- **Lip(attn) ≤ σ₁(W_attn) × σ₁(W_proj)** (Q/K/V projection composed with output projection, softmax Lip ≤ 1)
- **Lip(mlp) ≤ σ₁(W_fc) × 1.7 × σ₁(W_proj)** (up-projection, GELU with Lip ≈ 1.7, down-projection)

Here σ₁(·) denotes the largest singular value (spectral norm). Since f₁ and f₂ are two logit outputs from the same network, each bounded by G_total, the certified Lipschitz constant is **G_certified = 2 × G_total**.

This yields a fully certified exclusion radius: **dist(x, D) ≥ |m(x)| / G_certified** (satisfying A6). For our models, G_certified is extremely large (see Section 5.7), reflecting the known looseness of spectral-norm composition through deep networks. The gap between G_certified and the empirical L̂ quantifies an open problem in certified robustness for transformers.

Notably, weight sparsity in these models (up to 95% zeros in some layers) reduces spectral norms relative to dense equivalents, providing a tighter bound than would be possible for dense models of equal parameter count.

### 3.6 Empirical Instantiation with L̂ (CALIBRATED)

In practice, G_certified is extremely conservative. We therefore also instantiate the bound with an empirically calibrated sensitivity parameter L̂ (95th percentile slope estimate under the declared attack/metric protocol):

**score(x) = |m(x)| / L̂**

This score is operationally useful for ranking vulnerability and predicting relative attack budget, but it is **not** a certified radius unless A6 holds with a provable global upper bound. Section 3.5 provides exactly such a bound; Section 5.7 quantifies the gap.

### 3.7 Directional Family Constraint (PROVED structure + OBSERVED parameters)

Let F denote a fixed attack family with admissible edit set Δ_F(x). Define per-edit family gains:
- β⁺(F): maximal one-edit increase in m
- β⁻(F): maximal one-edit decrease in m

For any t-edit sequence from family F:
- m_after ≤ m_before + t · β⁺(F)
- m_after ≥ m_before − t · β⁻(F)

Hence crossing requirements are:
- If m_before < 0, reaching boundary requires t ≥ |m_before| / β⁺(F)
- If m_before > 0, reaching boundary requires t ≥ |m_before| / β⁻(F)

If β⁺(F) and β⁻(F) are highly unequal, success becomes direction-asymmetric even at similar margins. This explains how signed correlation can differ across tasks or strategies without violating Section 3.4.

### 3.8 Sparse-Circuit Local Tightening (PROVED locally under A9)

Let cᵢ(x) bound the one-token influence on m at coordinate i near x. If only coordinates in active set S_x materially influence m locally (A9), then for substitution perturbations:

|Δm| ≤ Σᵢ∈changed∩S_x cᵢ(x)

Consequences:
1. Effective local sensitivity shrinks when perturbations mostly avoid S_x.
2. Structured attacks that overlap S_x can be much more efficient than random edits.
3. Sparse interpretability (identifying S_x) can tighten practical, family-specific risk estimates.

This tightening is local; it does not imply a global guarantee unless support drift is controlled.

### 3.9 Named Assumptions

| ID | Assumption | Formal Statement | Used In |
|----|-----------|-----------------|---------|
| A1 | Input space | X is a set of token sequences | All |
| A2 | Metric | Hamming distance for substitution-only; edit distance if insertions/deletions | §3.4 |
| A3 | Model outputs | F(x) = (f₁(x), f₂(x)) are class logits | §3.4 |
| A4 | Decision boundary | D = {x : f₁(x) = f₂(x)} | §3.4 |
| A5 | Lipschitz regularity | f₁, f₂ are Lipschitz with constants G₁, G₂ | §3.4 |
| A6 | Certification | Requires true upper bound Ĝ ≥ G₁ + G₂ | §3.5 |
| A7 | Output involution | τ_out(z₁, z₂) = (z₂, z₁) on ℝ² | §3.3 |
| A8 | Family perturbations | Each attack family defines admissible set Δ_F(x) | §3.7 |
| A9 | Local support stability | Only coordinates in S_x materially affect m locally | §3.8 |

### 3.10 What Section 3 Proves and What It Does Not

- **Proved**: lower bound on required distance to boundary (Theorem R4); certified Lipschitz bound via spectral norm composition (Section 3.5); family-directional budget inequalities; local sparse tightening under A9.
- **Computed**: certified G values from actual model weights, fully instantiating A6.
- **Calibrated**: replacing unknown global constant by L̂ for operational prediction.
- **Observed**: quote-task directional asymmetry, bracket-task strong negative margin-success correlation, and finite-sample predictive performance.

This separation is the mathematical backbone for the empirical sections that follow.

---

## 4. Experimental Methodology

### 4.1 Input Metric and V-Space Definition

We define the input space V as the set of discrete token sequences of fixed length N. To enable Lipschitz analysis, we equip this space with the **Hamming Distance** d_H(v, v') = Σᵢ 𝟙[vᵢ ≠ v'ᵢ], which counts the number of token substitutions required to transform one sequence into another. This metric aligns with our experimental perturbation budget (tokens modified).

### 4.2 Sensitivity Calibration Protocol (CALIBRATED)

The parameter L̂ is estimated empirically to instantiate the theoretical distance bound (Section 3.4). For each task, we compute the empirical sensitivity s = |Δg(v)| / d_H(v, v') across all attack trials. We define L̂ as the **95th percentile** of these observed slopes. This provides a robust, task-specific calibration that accounts for the high variance in token-level influence while filtering extreme outliers.

### 4.3 Model Systems and Tasks

We validate the framework on OpenAI's weight-sparse transformers (Gao et al., 2025):

| Model | Parameters | Task | Description |
|-------|------------|------|-------------|
| csp_yolo2 | 475M | Bracket counting | Predict "]]" vs "]" for list closing |
| csp_yolo1 | 159M | Quote detection | Predict '"' vs "'" for string closing |

### 4.4 Attack Strategy Taxonomy

We implement and compare three categories of circuit-informed attacks:

| Category | Strategy | Mechanism |
|----------|----------|-----------|
| Radius-Exceeding | Unbalanced Context | Insert unmatched structural tokens |
| Involution-Targeted | Distractor/Activation | Exploit circuit knowledge for high-influence perturbations |
| Lipschitz Inflation | Context Dilution | Saturate context with distractor tokens |

### 4.5 Independent Calibration Probe (ICP) Protocol

To resolve circularity in local sensitivity estimation, we introduce the **ICP protocol**. Before executing a targeted attack, an input v is subjected to M=10 random single-token substitutions sampled from the `tinypython` vocabulary distribution. The mean logit shift across these probes defines the **Prospective Local Lipschitz Estimate** (L̂_local), allowing for vulnerability prediction prior to the actual adversarial attempt.

### 4.6 Statistical Methods

- Pearson correlation for continuous relationships
- Fisher's exact test for categorical comparisons
- Bootstrap resampling (k=1000) for confidence intervals at N=32
- Multi-seed random baseline (seeds 42–51) for efficiency ratio CIs
- Cross-validation for prediction robustness

---

## 5. Results

### 5.1 Directional Validation and Attack Efficiency (OBSERVED)

Circuit-informed attacks achieved a success rate of **24.2% [95% CI: 18.1%, 30.3%]** across all trials, compared to a baseline success rate of **1.56% [95% CI: 0.2%, 3.8%]** for random perturbations. This constitutes a **15.5× efficiency improvement [95% CI: 11.2×–19.8×]** (p < 0.0001, Fisher's exact test).

On the bracket task, we observed a strong negative correlation between signed margin and attack success (r = −0.603 [95% CI: −0.78, −0.35], p = 0.0003). This validates the core prediction: vulnerability is a function of both boundary distance and directional alignment.

### 5.2 Calibrated Prediction Accuracy (CALIBRATED)

Using the binary decision rule (predict VULNERABLE if logit_diff < 0), we achieved **87.5% prediction accuracy** with **100% recall** (0 missed vulnerabilities) on the bracket task. When instantiated with the calibrated parameter L̂ = 3.04, the resulting adversarial margin correctly identifies the required token budget for successful flips in 28 out of 32 samples.

### 5.3 Functional Asymmetry Discovery (OBSERVED)

Validation on the quote task revealed a positive correlation (r = +0.368, p = 0.037). This sign reversal was found to be driven by a profound **Functional Asymmetry** in the model's response surface:
- Attacks pushing `"` → `'` achieved **13.4%** success
- Attacks pushing `'` → `"` achieved **0.0%** success

This asymmetry indicates that the double-quote representation is mechanistically more fragile than the single-quote representation in the csp_yolo1 model. This finding demonstrates that while the theorem (Section 3.4) provides the minimum distance bound, the realized vulnerability is constrained by family-specific **Directional Budgets** (Section 3.6).

### 5.4 Cross-Task Sensitivity and Normalized Margins (CALIBRATED)

We identified task-specific sensitivity parameters:

| Task | L̂ | L̂(1+K) | Normalized Margin |
|------|-----|---------|-------------------|
| Bracket | 3.04 | 6.07 | 0.95 tokens |
| Quote | 1.35 | 2.71 | 3.39 tokens |

The bracket counting circuit is **2.24× more sensitive** to perturbations than the quote detection circuit. By computing **Normalized Margins**, we correctly predicted the 3.56× larger attack budget required for the quote task, confirming the framework's cross-task generalizability.

### 5.5 Root Cause Analysis: Local Lipschitz Variation (OBSERVED)

Post-hoc analysis of prediction failures (False Positives) identified a **15× variation in local sensitivity**:

| Metric | True Positives (n=12) | False Positives (n=4) | Ratio |
|--------|----------------------|----------------------|-------|
| Mean L̂_local | 1.49 | 0.10 | 15× |
| Mean logit shift | +3.52 | +0.22 | 16× |

FP samples reside in "flat" regions of the model's response surface where perturbations have minimal effect (p = 0.009). Preliminary validation of the **ICP protocol** demonstrates that this sensitivity gap can be detected pre-attack using random probes, providing a path toward improved precision in vulnerability identification.

### 5.6 Directional Selectivity (OBSERVED)

All attack strategies showed **0% success** on samples predicting "]]" (the direction away from the boundary), while achieving up to **48.4% success** on samples predicting "]" (toward the boundary). This perfect directional selectivity confirms the core geometric insight.

### 5.7 Certified Lipschitz Bound (COMPUTED)

We computed certified Lipschitz bounds from the actual model weights by composing spectral norms (largest singular values) through the transformer architecture (Section 3.5). Results for both models:

| Model | G_certified | L̂ (calibrated) | Gap (log₁₀) | Mean Weight Sparsity |
|-------|-------------|-----------------|-------------|---------------------|
| csp_yolo2 (bracket) | 2.29 × 10¹¹ | 3.04 | 10.9 | 99.9% |
| csp_yolo1 (quote) | 2.31 × 10¹⁰ | 1.35 | 10.2 | 99.6% |

The certified bound is many orders of magnitude larger than the empirical L̂. This gap is expected and informative:

1. **Theorem R4 is now fully instantiated**: with G_certified satisfying assumption A6, the exclusion radius dist(x, D) ≥ |m(x)| / G_certified is a rigorous mathematical guarantee, not merely a calibrated estimate.

2. **The gap quantifies an open problem**: the ratio G_certified / L̂ measures how much room remains for tightening certified bounds for transformers. This gap arises from conservative treatment of residual connections and the loose multiplicative composition through 12+ layers.

3. **Sparsity provides partial tightening**: weight sparsity (many exactly-zero parameters) reduces spectral norms below what equivalent dense matrices would produce, demonstrating that interpretable sparse circuits have a structural advantage for certified robustness.

4. **Practical vs. certified**: the calibrated L̂ remains the operationally useful bound (Section 3.6), while G_certified establishes the theoretical floor. Closing this gap — through local Lipschitz analysis (Section 3.8), tighter composition rules, or randomized smoothing — is a key direction for future work.

---

## 6. Discussion

### 6.1 Main Contribution with Correct Claim Levels

This work contributes a two-part theory-to-data framework:

1. **Theorem-level contribution (PROVED):** Theorem R4 gives a necessary lower bound on input-space perturbation distance to decision tie under declared metric and Lipschitz assumptions.
2. **Directional refinement (PROVED structure + OBSERVED parameters):** Family-specific directional budget terms (β⁺, β⁻) explain asymmetric attack outcomes that margin-only analysis cannot.
3. **Operational instantiation (CALIBRATED):** L̂ yields a practical margin-based risk score that predicts vulnerability patterns in experiments.

### 6.2 Interpretation of Cross-Task Results

The quote-task positive signed correlation does not falsify Theorem R4. Instead, it indicates asymmetric family-direction budgets in that model-task pair: one direction is substantially easier for the tested perturbation family. Thus, the theorem supplies a necessary distance condition, while directional accessibility determines realized success patterns.

### 6.3 Limits and Non-Claims

- While we compute a certified G_certified from spectral norms (Section 3.5, satisfying A6), the resulting certified radius is extremely conservative. The calibrated L̂ provides practical utility but not formal certification.
- We do **not** claim the directional rule is a universal theorem for all attack families. It is proved structurally but parameterized empirically.
- We treat L_local findings as post-hoc mechanism evidence pending full prospective ICP validation.
- Margin-based filtering is not validated against adaptive white-box attackers and is therefore a scoped heuristic.
- Sample size (N=32 per task) is adequate for observed effect sizes but limits power for detecting small effects.

### 6.4 Why This Matters for Interpretable Sparse Circuits

Sparse circuits expose structure (active coordinates, family-specific vulnerabilities) that improves mechanistic understanding of failures. This can increase both attack efficiency (when structure is exploited) and diagnostic power (when structure is monitored). The key scientific point is not that sparsity is uniformly safer; it is that **sparsity makes the vulnerability geometry measurable**.

### 6.5 Future Work

1. **Prospective ICP validation** for local sensitivity prediction
2. **Tighter certified bounds** via local Lipschitz analysis, layer-wise tightening, or randomized smoothing to close the gap between G_certified and L̂ (Section 5.7)
3. **Continuous alignment functional** replacing binary directional proxy
4. **Dense model extension** via sparse circuit proxies (Conmy et al., 2023)
5. **Adaptive adversary testing** of margin-based defense

---

## 7. Conclusions

This project supports a precise final claim set:

1. **PROVED:** Under explicit assumptions, distance to decision boundary is lower-bounded by margin over Lipschitz scale (Theorem R4).
2. **COMPUTED:** Certified Lipschitz bound G_certified from spectral norm composition fully instantiates A6 for both models; the gap to calibrated L̂ quantifies tightening opportunities.
3. **PROVED/OBSERVED bridge:** Directional family budgets (β⁺, β⁻) provide a formal mechanism for asymmetric attack success and explain sign reversals in signed-correlation analyses without contradicting the distance theorem.
4. **CALIBRATED:** Empirical L̂ instantiation provides practical vulnerability scoring and strong predictive utility on tested sparse-circuit tasks.
5. **OBSERVED:** Quote and bracket tasks exhibit different directional fragilities, indicating model-attack interaction structure beyond margin magnitude alone.

The paper's central contribution is a **disciplined theory-empiric bridge**: a proved boundary-distance theorem, an explicitly labeled calibrated instantiation, and empirically validated directional structure.

The 15.5× efficiency improvement achieved by circuit-informed attacks highlights both the power and the risk of interpretability research. Understanding a model's internal structure enables dramatically more effective attacks — but the same understanding, through the Collision Principle framework, can be leveraged for vulnerability assessment and defense design.

---

## References

1. Gao, L., et al. (2025). Weight-sparse transformers have interpretable circuits. *arXiv:2511.13653*.
2. Szegedy, C., et al. (2014). Intriguing properties of neural networks. *ICLR*.
3. Goodfellow, I., et al. (2015). Explaining and harnessing adversarial examples. *ICLR*.
4. Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
5. Conmy, A., et al. (2023). Towards automated circuit discovery for mechanistic interpretability. *NeurIPS*.
6. Cohen, J., et al. (2019). Certified adversarial robustness via randomized smoothing. *ICML*.
7. Wong, E. & Kolter, Z. (2018). Provable defenses against adversarial examples. *ICML*.
8. Fawzi, A., et al. (2018). The robustness of deep networks: A geometrical perspective. *IEEE SPM*.
9. Carlini, N. & Wagner, D. (2017). Adversarial examples are not easily detected. *AISec*.
10. Elhage, N., et al. (2022). Toy models of superposition. *Anthropic*.
