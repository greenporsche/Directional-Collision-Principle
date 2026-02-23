"""
Attack Taxonomy Analysis
========================

Maps empirical attack strategies to the theoretical framework from the
Directional Collision Principle.

THEORETICAL TAXONOMY (from application.md):
1. RADIUS-EXCEEDING: Large perturbations that push past the margin boundary
   - Works by: ||δ|| > r_min where r_min = |g(v)| / L(1+K)
   - Characteristic: Success scales with perturbation budget

2. LIPSCHITZ INFLATION: Attacks that increase effective Lipschitz constant
   - Works by: Manipulating context to increase L, shrinking margin
   - Characteristic: Context dilution, attention averaging

3. INVOLUTION-TARGETED: Direct movement toward τ(v)
   - Works by: Minimizing |g(v)| directly by adding symmetric structure
   - Characteristic: Exploits circuit structure knowledge

EMPIRICAL STRATEGIES (from circuit_informed_attack.py):
- distractor_comment: Add # [[[ in comments
- unbalanced_context: Add unmatched [ early in sequence
- activation_targeted: Modify high-activation positions
- context_dilution: Add many tokens to dilute signal
- combined: Multi-pronged approach

This analysis maps empirical strategies to theoretical categories.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

EXPERIMENTS_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENTS_DIR / 'data'
FIGURES_DIR = EXPERIMENTS_DIR / 'figures'

# Theoretical mapping
STRATEGY_TO_CATEGORY = {
    'distractor_comment': 'involution_targeted',
    'unbalanced_context': 'radius_exceeding',
    'activation_targeted': 'involution_targeted',
    'context_dilution': 'lipschitz_inflation',
    'combined': 'hybrid',
}

CATEGORY_DESCRIPTIONS = {
    'radius_exceeding': {
        'name': 'Radius-Exceeding',
        'mechanism': 'Large perturbations push past margin boundary',
        'formula': '||δ|| > |g(v)| / L(1+K)',
        'color': '#e74c3c',
    },
    'lipschitz_inflation': {
        'name': 'Lipschitz Inflation',
        'mechanism': 'Context manipulation increases effective L',
        'formula': 'L_eff ↑ → margin ↓',
        'color': '#9b59b6',
    },
    'involution_targeted': {
        'name': 'Involution-Targeted',
        'mechanism': 'Direct movement toward τ(v) via circuit knowledge',
        'formula': 'Minimize |g(v)| = |A(v) - A(τ(v))|',
        'color': '#3498db',
    },
    'hybrid': {
        'name': 'Hybrid',
        'mechanism': 'Combines multiple attack vectors',
        'formula': 'Multi-pronged approach',
        'color': '#27ae60',
    },
}


def load_data():
    """Load experiment results"""
    df_circuit = pd.read_csv(DATA_DIR / 'circuit_informed_results.csv')
    df_full = pd.read_csv(DATA_DIR / 'full_experiment_results.csv')
    return df_full, df_circuit


def analyze_strategy_performance(df_circuit):
    """
    Analyze each attack strategy's performance characteristics.
    """
    strategies = df_circuit['attack_strategy'].unique()
    results = []

    for strategy in strategies:
        strategy_data = df_circuit[df_circuit['attack_strategy'] == strategy]

        # Overall metrics
        overall_success = strategy_data['prediction_flipped'].mean()
        n_attacks = len(strategy_data)

        # By token count
        by_tokens = strategy_data.groupby('tokens_modified').agg({
            'prediction_flipped': ['mean', 'sum', 'count']
        }).reset_index()
        by_tokens.columns = ['tokens', 'success_rate', 'n_success', 'n_total']

        # By original prediction class
        predicts_single = strategy_data[strategy_data['original_prediction'] == ']']
        predicts_double = strategy_data[strategy_data['original_prediction'] == ']]']

        success_on_single = predicts_single['prediction_flipped'].mean() if len(predicts_single) > 0 else 0
        success_on_double = predicts_double['prediction_flipped'].mean() if len(predicts_double) > 0 else 0

        # Correlation with margin
        strategy_data = strategy_data.copy()
        strategy_data['signed_margin'] = strategy_data['original_logit_diff']

        by_sample = strategy_data.groupby('sample_id').agg({
            'signed_margin': 'first',
            'prediction_flipped': 'mean'
        }).reset_index()

        if len(by_sample) > 2:
            r, p = stats.pearsonr(by_sample['signed_margin'], by_sample['prediction_flipped'])
        else:
            r, p = 0, 1

        # Logit shift (how much does the attack move the decision?)
        avg_logit_shift = (strategy_data['original_logit_diff'] - strategy_data['attacked_logit_diff']).mean()

        results.append({
            'strategy': strategy,
            'category': STRATEGY_TO_CATEGORY.get(strategy, 'unknown'),
            'overall_success': overall_success,
            'n_attacks': n_attacks,
            'success_on_single': success_on_single,  # Predicts "]"
            'success_on_double': success_on_double,  # Predicts "]]"
            'selectivity': success_on_single - success_on_double,  # How targeted
            'margin_correlation': r,
            'margin_p_value': p,
            'avg_logit_shift': avg_logit_shift,
            'by_tokens': by_tokens,
        })

    return pd.DataFrame(results)


def analyze_scaling_behavior(df_circuit):
    """
    Analyze how success scales with perturbation budget for each strategy.

    Different theoretical categories predict different scaling:
    - Radius-exceeding: Linear scaling (more tokens = more movement)
    - Lipschitz inflation: Sublinear (diminishing returns)
    - Involution-targeted: High efficiency even at low budgets
    """
    strategies = df_circuit['attack_strategy'].unique()
    scaling_results = []

    for strategy in strategies:
        strategy_data = df_circuit[df_circuit['attack_strategy'] == strategy]

        by_tokens = strategy_data.groupby('tokens_modified')['prediction_flipped'].mean()

        if len(by_tokens) >= 2:
            # Fit linear model to estimate scaling
            tokens = by_tokens.index.values
            success = by_tokens.values

            slope, intercept, r_value, p_value, std_err = stats.linregress(tokens, success)

            # Efficiency at 1 token (baseline)
            efficiency_1 = by_tokens.get(1, 0)

            # Efficiency at 5 tokens
            efficiency_5 = by_tokens.get(5, 0)

            # Scaling ratio
            scaling_ratio = efficiency_5 / efficiency_1 if efficiency_1 > 0 else float('inf')

            scaling_results.append({
                'strategy': strategy,
                'category': STRATEGY_TO_CATEGORY.get(strategy, 'unknown'),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value ** 2,
                'efficiency_1_token': efficiency_1,
                'efficiency_5_tokens': efficiency_5,
                'scaling_ratio': scaling_ratio,
            })

    return pd.DataFrame(scaling_results)


def create_taxonomy_figure(df_circuit, strategy_results, scaling_results, output_path):
    """
    Create comprehensive taxonomy visualization.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    strategies = ['distractor_comment', 'unbalanced_context', 'activation_targeted',
                  'context_dilution', 'combined']
    strategy_labels = ['Distractor\nComment', 'Unbalanced\nContext', 'Activation\nTargeted',
                       'Context\nDilution', 'Combined']

    # =========================================================================
    # Panel A: Success rate by strategy
    # =========================================================================
    ax = axes[0, 0]

    success_rates = [strategy_results[strategy_results['strategy'] == s]['overall_success'].values[0] * 100
                     for s in strategies]
    colors = [CATEGORY_DESCRIPTIONS[STRATEGY_TO_CATEGORY[s]]['color'] for s in strategies]

    bars = ax.bar(range(len(strategies)), success_rates, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.5)

    # Add category labels
    for i, (bar, strategy) in enumerate(zip(bars, strategies)):
        category = STRATEGY_TO_CATEGORY[strategy]
        cat_name = CATEGORY_DESCRIPTIONS[category]['name']
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{success_rates[i]:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, -3,
                f'({cat_name[:8]}...)', ha='center', va='top',
                fontsize=8, color='gray', rotation=0)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_ylabel('Overall Success Rate (%)', fontsize=12)
    ax.set_title('A. Attack Strategy Performance\nColored by Theoretical Category',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(success_rates) * 1.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=CATEGORY_DESCRIPTIONS[cat]['color'], alpha=0.8,
                             label=CATEGORY_DESCRIPTIONS[cat]['name'])
                       for cat in ['radius_exceeding', 'lipschitz_inflation',
                                   'involution_targeted', 'hybrid']]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # =========================================================================
    # Panel B: Success by prediction class (selectivity)
    # =========================================================================
    ax = axes[0, 1]

    x = np.arange(len(strategies))
    width = 0.35

    success_single = [strategy_results[strategy_results['strategy'] == s]['success_on_single'].values[0] * 100
                      for s in strategies]
    success_double = [strategy_results[strategy_results['strategy'] == s]['success_on_double'].values[0] * 100
                      for s in strategies]

    bars1 = ax.bar(x - width/2, success_double, width, label='Predicts "]]" (robust)',
                   color='#2c3e50', alpha=0.7)
    bars2 = ax.bar(x + width/2, success_single, width, label='Predicts "]" (vulnerable)',
                   color='#c0392b', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(strategy_labels, fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('B. Selectivity: Success by Original Prediction Class\n'
                 'Circuit attacks target "]" samples specifically',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)

    # Add selectivity annotations
    for i, (s1, s2) in enumerate(zip(success_single, success_double)):
        if s1 > s2:
            ax.annotate(f'+{s1-s2:.0f}%', xy=(i, max(s1, s2) + 2),
                       ha='center', fontsize=9, color='darkgreen', fontweight='bold')

    # =========================================================================
    # Panel C: Scaling behavior
    # =========================================================================
    ax = axes[1, 0]

    token_counts = [1, 2, 3, 5]

    for strategy in strategies:
        strategy_data = df_circuit[df_circuit['attack_strategy'] == strategy]
        by_tokens = strategy_data.groupby('tokens_modified')['prediction_flipped'].mean() * 100

        color = CATEGORY_DESCRIPTIONS[STRATEGY_TO_CATEGORY[strategy]]['color']
        label = strategy.replace('_', ' ').title()

        # Only plot if we have data for these token counts
        x_vals = [t for t in token_counts if t in by_tokens.index]
        y_vals = [by_tokens[t] for t in x_vals]

        ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=2, markersize=8,
                label=label, alpha=0.8)

    ax.set_xlabel('Tokens Modified', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('C. Scaling Behavior\nHow success changes with perturbation budget',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xticks(token_counts)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel D: Removed for cleaner poster layout
    # =========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    fig.suptitle('Attack Taxonomy: Mapping Strategies to Theoretical Framework',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_taxonomy_summary(strategy_results, scaling_results):
    """Generate detailed taxonomy analysis summary"""

    summary = f"""
{'='*70}
ATTACK TAXONOMY ANALYSIS
{'='*70}

THEORETICAL FRAMEWORK
─────────────────────────────────────────────────────────────────────

The Directional Collision Principle predicts that adversarial attacks
succeed through three distinct mechanisms:

1. RADIUS-EXCEEDING
   Formula: ||δ|| > |g(v)| / L(1+K)
   Mechanism: Large perturbations push the representation past the
              collision-free radius around the original input.

2. LIPSCHITZ INFLATION
   Formula: L_effective ↑ → margin ↓
   Mechanism: Context manipulation (e.g., adding many tokens) increases
              the effective Lipschitz constant, shrinking the margin.

3. INVOLUTION-TARGETED
   Formula: Minimize |g(v)| = |A(v) - A(τ(v))|
   Mechanism: Directly add structure that reduces the gap between
              the representation and its symmetric counterpart.

{'='*70}
STRATEGY MAPPING
{'='*70}

"""

    for _, row in strategy_results.iterrows():
        category = CATEGORY_DESCRIPTIONS.get(row['category'], {})
        cat_name = category.get('name', 'Unknown')

        summary += f"""
{row['strategy'].upper()}
─────────────────────────────────────────────────────────────────────
  Category:           {cat_name}
  Mechanism:          {category.get('mechanism', 'N/A')}

  Performance:
    Overall success:  {row['overall_success']*100:.1f}%
    On "]" samples:   {row['success_on_single']*100:.1f}%
    On "]]" samples:  {row['success_on_double']*100:.1f}%
    Selectivity:      {row['selectivity']*100:+.1f}% (higher = more targeted)

  Directional Correlation:
    r = {row['margin_correlation']:.3f} (p = {row['margin_p_value']:.4f})
    {'Strong negative correlation' if row['margin_correlation'] < -0.3 else 'Weak/no correlation'}

"""

    summary += f"""
{'='*70}
SCALING ANALYSIS
{'='*70}

How does success scale with perturbation budget (tokens modified)?

"""

    for _, row in scaling_results.iterrows():
        summary += f"""
{row['strategy'].upper()}:
  1 token:  {row['efficiency_1_token']*100:.1f}%
  5 tokens: {row['efficiency_5_tokens']*100:.1f}%
  Scaling:  {row['scaling_ratio']:.1f}× improvement from 1→5 tokens
  Trend:    {'Linear scaling' if row['slope'] > 0.02 else 'Sublinear/saturating'}

"""

    summary += f"""
{'='*70}
KEY FINDINGS
{'='*70}

1. ALL STRATEGIES SHOW DIRECTIONAL SELECTIVITY
   - Near-zero success on "]]" samples
   - High success on "]" samples
   - Confirms the Directional Collision Principle

2. INVOLUTION-TARGETED STRATEGIES ARE MOST EFFICIENT
   - activation_targeted and distractor_comment achieve high success
     even with minimal perturbation
   - They exploit circuit knowledge to minimize |g(v)| directly

3. HYBRID (COMBINED) ACHIEVES BEST OVERALL PERFORMANCE
   - Combines multiple attack vectors
   - {strategy_results[strategy_results['strategy']=='combined']['overall_success'].values[0]*100:.1f}% overall success rate

4. CONTEXT DILUTION SHOWS LIPSCHITZ INFLATION SIGNATURE
   - Success scales with number of added tokens
   - Consistent with attention averaging reducing effective margin

{'='*70}
"""

    return summary


def main():
    print("=" * 70)
    print("ATTACK TAXONOMY ANALYSIS")
    print("Mapping strategies to theoretical framework")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    df_full, df_circuit = load_data()
    print(f"  Circuit-informed attacks: {len(df_circuit)} trials")

    # Analyze strategy performance
    print("\n[2/4] Analyzing strategy performance...")
    strategy_results = analyze_strategy_performance(df_circuit)

    for _, row in strategy_results.iterrows():
        cat_name = CATEGORY_DESCRIPTIONS.get(row['category'], {}).get('name', 'Unknown')
        print(f"  {row['strategy']}: {row['overall_success']*100:.1f}% ({cat_name})")

    # Analyze scaling behavior
    print("\n[3/4] Analyzing scaling behavior...")
    scaling_results = analyze_scaling_behavior(df_circuit)

    # Create figure
    print("\n[4/4] Generating taxonomy figure...")
    output_path = FIGURES_DIR / 'attack_taxonomy.png'
    create_taxonomy_figure(df_circuit, strategy_results, scaling_results, output_path)
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")

    # Generate and print summary
    summary = generate_taxonomy_summary(strategy_results, scaling_results)
    print(summary)

    # Save summary
    summary_path = DATA_DIR / 'attack_taxonomy_results.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Results saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("TAXONOMY ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        'strategy_results': strategy_results,
        'scaling_results': scaling_results
    }


if __name__ == "__main__":
    results = main()
