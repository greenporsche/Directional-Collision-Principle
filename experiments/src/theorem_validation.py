"""
Theorem Validation: Does the Collision Principle predict adversarial vulnerability?
===================================================================================

HYPOTHESIS: The Adversarial Margin |g(v)| = |logit_diff| predicts attack success.
            Samples with LOWER margin should be MORE vulnerable to attack.

This directly validates the core prediction of the Collision Principle theorem:
    dist(v, Z) >= |g(v)| / L(1+K)

If the theorem is correct:
- Low |g(v)| -> small exclusion radius -> easy to attack
- High |g(v)| -> large exclusion radius -> hard to attack

Therefore: negative correlation between |logit_diff| and attack success.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import expit  # logistic function
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

EXPERIMENTS_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENTS_DIR / 'data'
FIGURES_DIR = EXPERIMENTS_DIR / 'figures'


def load_data():
    """Load experiment results"""
    df_full = pd.read_csv(DATA_DIR / 'full_experiment_results.csv')
    df_circuit = pd.read_csv(DATA_DIR / 'circuit_informed_results.csv')
    return df_full, df_circuit


def compute_sample_metrics(df):
    """
    For each sample, compute:
    - margin: |original_logit_diff| (the adversarial margin from theorem)
    - success_rate: fraction of attacks that flipped the prediction
    - n_attacks: number of attack attempts

    The key insight: margin IS |g(v)| = |A(v) - A(τ(v))| from the theorem!
    """
    # Get clean (baseline) data for each sample
    clean = df[df['attack_type'] == 'clean'][['sample_id', 'original_logit_diff', 'original_prediction']].copy()
    clean['margin'] = clean['original_logit_diff'].abs()

    # Get attack outcomes for each sample
    attacks = df[df['attack_type'] != 'clean'].copy()

    # Compute success rate per sample (across all attack types and token counts)
    attack_summary = attacks.groupby('sample_id').agg({
        'prediction_flipped': ['sum', 'count', 'mean']
    }).reset_index()
    attack_summary.columns = ['sample_id', 'n_flipped', 'n_attacks', 'success_rate']

    # Merge
    samples = clean.merge(attack_summary, on='sample_id')

    return samples


def pearson_correlation_analysis(samples):
    """
    Primary validation: Pearson correlation between margin and success rate.

    Expected: r < 0 (negative correlation)
              p < 0.05 (statistically significant)
    """
    r, p = stats.pearsonr(samples['margin'], samples['success_rate'])

    return {
        'pearson_r': r,
        'p_value': p,
        'n_samples': len(samples),
        'significant': p < 0.05,
        'direction_correct': r < 0
    }


def spearman_correlation_analysis(samples):
    """
    Robustness check: Spearman rank correlation (non-parametric)
    """
    rho, p = stats.spearmanr(samples['margin'], samples['success_rate'])

    return {
        'spearman_rho': rho,
        'p_value': p,
        'significant': p < 0.05
    }


def logistic_regression_analysis(samples):
    """
    Fit logistic regression: P(attack_success) ~ f(margin)

    This gives us an interpretable model:
    - Coefficient tells us how much margin reduces attack probability
    - Can be used for prediction
    """
    from scipy.optimize import minimize

    # Prepare data
    X = samples['margin'].values
    y = samples['success_rate'].values

    # Simple linear regression for success rate (not truly logistic, but interpretable)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err
    }


def point_biserial_analysis(df):
    """
    For individual attack trials: point-biserial correlation
    between margin and binary success.
    """
    attacks = df[df['attack_type'] != 'clean'].copy()
    attacks['margin'] = attacks['original_logit_diff'].abs()

    r, p = stats.pointbiserialr(attacks['prediction_flipped'], attacks['margin'])

    return {
        'point_biserial_r': r,
        'p_value': p,
        'n_trials': len(attacks),
        'significant': p < 0.05
    }


def create_main_validation_figure(samples, pearson_results, output_path):
    """
    FIGURE 1: The Core Theorem Validation

    Shows: margin vs success rate with correlation
    This is THE key figure proving the theorem works.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by original prediction type
    colors = ['#e74c3c' if p == ']]' else '#3498db' for p in samples['original_prediction']]

    # Scatter plot
    scatter = ax.scatter(
        samples['margin'],
        samples['success_rate'] * 100,
        c=colors,
        s=150,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.5
    )

    # Regression line
    z = np.polyfit(samples['margin'], samples['success_rate'] * 100, 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(samples['margin'].min() * 0.9, samples['margin'].max() * 1.1, 100)
    ax.plot(x_line, p_line(x_line), 'k--', linewidth=2.5, alpha=0.8,
            label=f'Linear fit')

    # Add confidence band (approximate)
    y_pred = p_line(samples['margin'])
    residuals = samples['success_rate'] * 100 - y_pred
    se = np.std(residuals) * np.sqrt(1 + 1/len(samples))
    ax.fill_between(x_line, p_line(x_line) - 1.96*se, p_line(x_line) + 1.96*se,
                    alpha=0.15, color='gray', label='95% CI')

    # Labels and title
    ax.set_xlabel('Adversarial Margin |g(v)| = |logit("]]) - logit("]")|', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')

    r = pearson_results['pearson_r']
    p = pearson_results['p_value']
    title = f'THEOREM VALIDATION: Collision Principle Predicts Vulnerability\n'
    title += f'r = {r:.3f}, p = {p:.2e} ({"significant" if p < 0.05 else "not significant"})'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Legend for prediction types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=12, label='Nested list samples (predict "]]")'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=12, label='Flat list samples (predict "]")'),
        Line2D([0], [0], linestyle='--', color='black', linewidth=2, label='Linear fit'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Add theorem annotation box
    theorem_text = (
        "Collision Principle:\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "dist(v, Z) ≥ |g(v)| / L(1+K)\n\n"
        "Prediction:\n"
        "Low |g(v)| → Small safe zone\n"
        "           → Easy to attack\n\n"
        f"Result: r = {r:.3f}\n"
        f"{'✓ VALIDATED' if r < 0 and p < 0.05 else '✗ NOT VALIDATED'}"
    )
    ax.text(0.02, 0.98, theorem_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='orange', alpha=0.9, linewidth=2))

    # Style
    ax.set_xlim(samples['margin'].min() * 0.8, samples['margin'].max() * 1.1)
    ax.set_ylim(-5, max(samples['success_rate'] * 100) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_by_token_figure(df, output_path):
    """
    FIGURE 2: Margin effect across different perturbation budgets

    Shows: Does the margin-vulnerability relationship hold at all token counts?
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    token_counts = [1, 2, 3, 5]

    for ax, n_tokens in zip(axes, token_counts):
        # Filter to this token count
        attacks = df[(df['attack_type'] != 'clean') & (df['tokens_modified'] == n_tokens)].copy()
        attacks['margin'] = attacks['original_logit_diff'].abs()

        # Get success rate per sample at this token count
        sample_success = attacks.groupby('sample_id').agg({
            'margin': 'first',
            'prediction_flipped': 'mean'
        }).reset_index()

        if len(sample_success) > 2:
            r, p = stats.pearsonr(sample_success['margin'], sample_success['prediction_flipped'])

            # Scatter
            colors = ['#e74c3c' if s > 0 else '#3498db' for s in sample_success['prediction_flipped']]
            ax.scatter(sample_success['margin'], sample_success['prediction_flipped'] * 100,
                      c=colors, s=80, alpha=0.7, edgecolors='black')

            # Regression line
            z = np.polyfit(sample_success['margin'], sample_success['prediction_flipped'] * 100, 1)
            p_line = np.poly1d(z)
            x_range = np.linspace(sample_success['margin'].min(), sample_success['margin'].max(), 50)
            ax.plot(x_range, p_line(x_range), 'k--', linewidth=2)

            ax.set_title(f'{n_tokens} Token(s) Modified\nr={r:.3f}, p={p:.3f}', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'{n_tokens} Token(s) Modified\n(insufficient data)', fontsize=12)

        ax.set_xlabel('Margin |g(v)|', fontsize=10)
        ax.set_ylabel('Success Rate (%)', fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Theorem Validation Across Perturbation Budgets\n'
                 'Negative correlation should hold regardless of attack strength',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_attack_type_comparison(df_full, df_circuit, output_path):
    """
    FIGURE 3: Does the margin effect differ by attack type?

    Compares: random vs circuit-informed attacks
    Theory predicts: circuit attacks should be more effective at LOW margin samples
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Random attacks
    random_attacks = df_full[df_full['attack_type'] == 'random'].copy()
    random_attacks['margin'] = random_attacks['original_logit_diff'].abs()

    random_by_sample = random_attacks.groupby('sample_id').agg({
        'margin': 'first',
        'prediction_flipped': 'mean'
    }).reset_index()

    ax1.scatter(random_by_sample['margin'], random_by_sample['prediction_flipped'] * 100,
                s=100, alpha=0.7, c='#3498db', edgecolors='black')

    if len(random_by_sample) > 2:
        r, p = stats.pearsonr(random_by_sample['margin'], random_by_sample['prediction_flipped'])
        z = np.polyfit(random_by_sample['margin'], random_by_sample['prediction_flipped'] * 100, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(random_by_sample['margin'].min(), random_by_sample['margin'].max(), 50)
        ax1.plot(x_range, p_line(x_range), 'k--', linewidth=2)
        ax1.set_title(f'Random Attacks\nr={r:.3f}, p={p:.3f}', fontsize=14, fontweight='bold')

    ax1.set_xlabel('Margin |g(v)|', fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Circuit-informed attacks (combined strategy)
    circuit_attacks = df_circuit[df_circuit['attack_strategy'] == 'combined'].copy()
    circuit_attacks['margin'] = circuit_attacks['original_logit_diff'].abs()

    circuit_by_sample = circuit_attacks.groupby('sample_id').agg({
        'margin': 'first',
        'prediction_flipped': 'mean'
    }).reset_index()

    ax2.scatter(circuit_by_sample['margin'], circuit_by_sample['prediction_flipped'] * 100,
                s=100, alpha=0.7, c='#e74c3c', edgecolors='black')

    if len(circuit_by_sample) > 2:
        r, p = stats.pearsonr(circuit_by_sample['margin'], circuit_by_sample['prediction_flipped'])
        z = np.polyfit(circuit_by_sample['margin'], circuit_by_sample['prediction_flipped'] * 100, 1)
        p_line = np.poly1d(z)
        x_range = np.linspace(circuit_by_sample['margin'].min(), circuit_by_sample['margin'].max(), 50)
        ax2.plot(x_range, p_line(x_range), 'k--', linewidth=2)
        ax2.set_title(f'Circuit-Informed Attacks\nr={r:.3f}, p={p:.3f}', fontsize=14, fontweight='bold')

    ax2.set_xlabel('Margin |g(v)|', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Margin-Vulnerability Correlation by Attack Type\n'
                 'Theorem should hold for both, but circuit attacks exploit it better',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_results_summary(samples, pearson, spearman, logistic, point_biserial):
    """Generate comprehensive results summary"""

    summary = f"""
{'='*70}
THEOREM VALIDATION RESULTS: COLLISION PRINCIPLE
{'='*70}

RESEARCH QUESTION
-----------------
Does the Adversarial Margin |g(v)| = |logit_diff| predict attack vulnerability?

HYPOTHESIS
----------
According to the Collision Principle theorem:
    dist(v, Z) >= |g(v)| / L(1+K)

Samples with LOW |g(v)| have smaller "exclusion radius" and should be
MORE vulnerable to adversarial attack.

Expected: NEGATIVE correlation between margin and attack success.

{'='*70}
STATISTICAL RESULTS
{'='*70}

1. PEARSON CORRELATION (Primary)
   -----------------------------
   r = {pearson['pearson_r']:.4f}
   p = {pearson['p_value']:.6f}
   n = {pearson['n_samples']} samples

   Interpretation:
   - Direction: {'CORRECT (negative)' if pearson['direction_correct'] else 'INCORRECT (positive)'}
   - Significance: {'YES (p < 0.05)' if pearson['significant'] else 'NO (p >= 0.05)'}
   - Verdict: {'✓ THEOREM VALIDATED' if pearson['direction_correct'] and pearson['significant'] else '✗ THEOREM NOT VALIDATED'}

2. SPEARMAN CORRELATION (Robustness Check)
   ---------------------------------------
   rho = {spearman['spearman_rho']:.4f}
   p = {spearman['p_value']:.6f}

   (Non-parametric, robust to outliers)
   Significant: {'YES' if spearman['significant'] else 'NO'}

3. LINEAR REGRESSION
   -----------------
   slope = {logistic['slope']:.6f}
   intercept = {logistic['intercept']:.4f}
   R² = {logistic['r_squared']:.4f}
   p = {logistic['p_value']:.6f}

   Interpretation:
   For each 1-unit increase in margin, success rate changes by {logistic['slope']*100:.2f}%

4. POINT-BISERIAL (Individual Trials)
   ----------------------------------
   r = {point_biserial['point_biserial_r']:.4f}
   p = {point_biserial['p_value']:.6f}
   n = {point_biserial['n_trials']} trials

   (Correlation between margin and binary success for each attack)

{'='*70}
KEY FINDINGS
{'='*70}

1. The Collision Principle's prediction is {'SUPPORTED' if pearson['direction_correct'] else 'NOT SUPPORTED'}:
   - Lower margin samples ARE {'more' if pearson['direction_correct'] else 'NOT more'} vulnerable
   - Correlation coefficient: r = {pearson['pearson_r']:.3f}

2. Statistical significance: {'ACHIEVED' if pearson['significant'] else 'NOT ACHIEVED'} (p = {pearson['p_value']:.4f})

3. Effect size interpretation (|r|):
   - |r| < 0.1: negligible
   - 0.1 <= |r| < 0.3: small
   - 0.3 <= |r| < 0.5: medium
   - |r| >= 0.5: large

   This study: |r| = {abs(pearson['pearson_r']):.3f} = {'large' if abs(pearson['pearson_r']) >= 0.5 else 'medium' if abs(pearson['pearson_r']) >= 0.3 else 'small' if abs(pearson['pearson_r']) >= 0.1 else 'negligible'}

{'='*70}
CONCLUSION
{'='*70}

{'The Collision Principle theorem is VALIDATED by this experiment.' if pearson['direction_correct'] and pearson['significant'] else 'The Collision Principle theorem could not be validated with statistical significance.'}

The Adversarial Margin |g(v)| = |logit_diff| {'successfully predicts' if pearson['significant'] else 'does not significantly predict'}
which samples will be vulnerable to adversarial attack.

{'='*70}
"""
    return summary


def main():
    print("=" * 70)
    print("THEOREM VALIDATION: COLLISION PRINCIPLE")
    print("Does Adversarial Margin predict attack vulnerability?")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading experiment data...")
    df_full, df_circuit = load_data()
    print(f"  Full experiment: {len(df_full)} rows")
    print(f"  Circuit-informed: {len(df_circuit)} rows")

    # Compute sample metrics
    print("\n[2/6] Computing sample-level metrics...")
    samples = compute_sample_metrics(df_full)
    print(f"  Processed {len(samples)} samples")
    print(f"  Margin range: [{samples['margin'].min():.2f}, {samples['margin'].max():.2f}]")
    print(f"  Success rate range: [{samples['success_rate'].min():.1%}, {samples['success_rate'].max():.1%}]")

    # Statistical analyses
    print("\n[3/6] Running statistical analyses...")
    pearson = pearson_correlation_analysis(samples)
    spearman = spearman_correlation_analysis(samples)
    logistic = logistic_regression_analysis(samples)
    point_biserial = point_biserial_analysis(df_full)

    print(f"  Pearson r = {pearson['pearson_r']:.4f} (p = {pearson['p_value']:.4f})")
    print(f"  Spearman rho = {spearman['spearman_rho']:.4f} (p = {spearman['p_value']:.4f})")

    # Generate figures
    print("\n[4/6] Generating main validation figure...")
    fig1_path = create_main_validation_figure(samples, pearson, FIGURES_DIR / 'theorem_validation_main.png')
    print(f"  Saved: {fig1_path}")

    print("\n[5/6] Generating token-count breakdown figure...")
    fig2_path = create_by_token_figure(df_full, FIGURES_DIR / 'theorem_validation_by_tokens.png')
    print(f"  Saved: {fig2_path}")

    print("\n[6/6] Generating attack-type comparison figure...")
    fig3_path = create_attack_type_comparison(df_full, df_circuit, FIGURES_DIR / 'theorem_validation_by_attack.png')
    print(f"  Saved: {fig3_path}")

    # Print results summary
    summary = generate_results_summary(samples, pearson, spearman, logistic, point_biserial)
    print(summary)

    # Save summary to file
    summary_path = DATA_DIR / 'theorem_validation_results.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Results saved to: {summary_path}")

    # Final verdict
    print("\n" + "=" * 70)
    if pearson['direction_correct'] and pearson['significant']:
        print("✓ THEOREM VALIDATED")
        print("  The Collision Principle correctly predicts adversarial vulnerability.")
        print(f"  Correlation: r = {pearson['pearson_r']:.3f}, p = {pearson['p_value']:.4f}")
    else:
        print("✗ VALIDATION INCONCLUSIVE")
        print("  More data or different analysis may be needed.")
    print("=" * 70)

    return {
        'samples': samples,
        'pearson': pearson,
        'spearman': spearman,
        'logistic': logistic,
        'point_biserial': point_biserial
    }


if __name__ == "__main__":
    results = main()
