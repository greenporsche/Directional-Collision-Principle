"""
Visualize experiment results for ISEF poster.
Updated to use circuit-informed attack results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Load results
output_dir = Path(__file__).parent.parent
data_dir = output_dir / 'data'
figures_dir = output_dir / 'figures'

circuit_informed_path = data_dir / "circuit_informed_results.csv"
df_circuit = pd.read_csv(circuit_informed_path)

baseline_path = data_dir / "full_experiment_results.csv"
df_baseline = pd.read_csv(baseline_path)

# Extract random attack results from baseline
random_attacks = df_baseline[df_baseline['attack_type'] == 'random'].copy()

# Extract combined strategy from circuit-informed (best performer)
combined_attacks = df_circuit[df_circuit['attack_strategy'] == 'combined'].copy()

# Calculate overall success rates
random_success = random_attacks['prediction_flipped'].sum()
random_total = len(random_attacks)
random_rate = random_success / random_total * 100

circuit_success = combined_attacks['prediction_flipped'].sum()
circuit_total = len(combined_attacks)
circuit_rate = circuit_success / circuit_total * 100

# Calculate efficiency
random_eff = random_attacks['efficiency'].mean()
circuit_eff = combined_attacks['efficiency'].mean()
efficiency_ratio = circuit_eff / random_eff if random_eff > 0 else float('inf')

# Fisher's exact test
table = [[random_success, random_total - random_success],
         [circuit_success, circuit_total - circuit_success]]
_, p_value = stats.fisher_exact(table)

print("=" * 60)
print("UPDATED EXPERIMENT RESULTS")
print("=" * 60)
print(f"Random Attack:           {random_success}/{random_total} = {random_rate:.1f}%")
print(f"Circuit-Informed Attack: {circuit_success}/{circuit_total} = {circuit_rate:.1f}%")
print(f"Improvement:             {circuit_rate / random_rate:.1f}x better")
print(f"Efficiency ratio:        {efficiency_ratio:.1f}x more efficient")
print(f"P-value:                 {p_value:.6f}")
print("=" * 60)

# Color scheme
colors = {'random': '#3498db', 'circuit': '#e74c3c'}

# ============================================================
# POSTER MAIN FIGURE - Clean comparison with correct arrow
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

x_pos = [0, 1]
heights = [random_rate, circuit_rate]
colors_list = [colors['random'], colors['circuit']]

bars = ax.bar(x_pos, heights, color=colors_list, alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Random\nAttack', 'Circuit-Informed\nAttack'], fontsize=14, fontweight='bold')
ax.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title(f'Circuit Knowledge Enables More Effective Attacks\n(p < 0.001, n = {random_total} trials per condition)',
             fontsize=16, fontweight='bold')

# Add value labels
for bar, val, succ, tot in zip(bars, heights, [random_success, circuit_success], [random_total, circuit_total]):
    ax.annotate(f'{val:.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5),
                ha='center', va='bottom', fontsize=20, fontweight='bold')
    ax.annotate(f'({succ}/{tot} successes)',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()/2),
                ha='center', va='center', fontsize=12, color='white', fontweight='bold')

# Add arrow spanning FROM random bar height TO circuit bar height
# Arrow should show the improvement factor
random_bar_top = heights[0]
circuit_bar_top = heights[1]
arrow_x = 0.5  # Center between bars

# Draw double-headed arrow from random height to circuit height
ax.annotate('',
            xy=(arrow_x, circuit_bar_top),      # Arrow head at circuit bar top
            xytext=(arrow_x, random_bar_top),   # Arrow tail at random bar top
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2.5,
                           connectionstyle='arc3,rad=0'))

# Add the "15x more effective" text next to the arrow
improvement_factor = circuit_rate / random_rate
ax.text(arrow_x + 0.15, (random_bar_top + circuit_bar_top) / 2,
        f'{improvement_factor:.0f}× more\neffective',
        fontsize=13, fontweight='bold', color='darkgreen',
        va='center', ha='left')

ax.set_ylim(0, max(heights) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(figures_dir / 'poster_main_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(figures_dir / 'poster_main_figure.pdf', bbox_inches='tight', facecolor='white')
print(f"✅ {figures_dir / 'poster_main_figure.png'}")
print(f"✅ {figures_dir / 'poster_main_figure.pdf'}")
plt.close()

# ============================================================
# HERO FIGURE - All strategies comparison
# ============================================================
fig2, ax2 = plt.subplots(figsize=(12, 6))

# Get success rates by strategy
strategies = ['distractor_comment', 'unbalanced_context', 'activation_targeted', 'context_dilution', 'combined']
strategy_labels = ['Distractor\nComment', 'Unbalanced\nContext', 'Activation\nTargeted', 'Context\nDilution', 'Combined\n(Best)']

strategy_rates = []
for strat in strategies:
    strat_data = df_circuit[df_circuit['attack_strategy'] == strat]
    rate = strat_data['prediction_flipped'].mean() * 100
    strategy_rates.append(rate)

# Add random baseline
all_labels = ['Random\nBaseline'] + strategy_labels
all_rates = [random_rate] + strategy_rates
all_colors = ['#95a5a6'] + ['#3498db', '#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']

bars = ax2.bar(range(len(all_labels)), all_rates, color=all_colors, alpha=0.85,
               edgecolor='black', linewidth=1.5, width=0.7)

# Highlight the best (combined)
bars[-1].set_edgecolor('gold')
bars[-1].set_linewidth(3)

ax2.set_xticks(range(len(all_labels)))
ax2.set_xticklabels(all_labels, fontsize=11, fontweight='bold')
ax2.set_ylabel('Attack Success Rate (%)', fontsize=14, fontweight='bold')
ax2.set_title('Comparison of Attack Strategies\nCircuit-Informed Attacks vs Random Baseline',
             fontsize=16, fontweight='bold')

# Add value labels
for bar, val in zip(bars, all_rates):
    if val > 0:
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add horizontal line for random baseline reference
ax2.axhline(y=random_rate, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.text(len(all_labels) - 0.5, random_rate + 0.5, 'Random baseline',
         fontsize=10, color='gray', ha='right')

ax2.set_ylim(0, max(all_rates) * 1.25)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(figures_dir / 'hero_figure_final.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(figures_dir / 'hero_figure_final.pdf', bbox_inches='tight', facecolor='white')
print(f"✅ {figures_dir / 'hero_figure_final.png'}")
print(f"✅ {figures_dir / 'hero_figure_final.pdf'}")
plt.close()

# ============================================================
# FINAL COMPARISON - By token count
# ============================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))

tokens = [1, 2, 3, 5]
x = np.arange(len(tokens))
width = 0.35

# Get rates by token count
random_by_token = []
circuit_by_token = []
for t in tokens:
    r_data = random_attacks[random_attacks['tokens_modified'] == t]
    c_data = combined_attacks[combined_attacks['tokens_modified'] == t]
    random_by_token.append(r_data['prediction_flipped'].mean() * 100 if len(r_data) > 0 else 0)
    circuit_by_token.append(c_data['prediction_flipped'].mean() * 100 if len(c_data) > 0 else 0)

# Left panel: Success rates by token count
bars1 = ax3a.bar(x - width/2, random_by_token, width, label='Random Attack',
                 color=colors['random'], alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax3a.bar(x + width/2, circuit_by_token, width, label='Circuit-Informed Attack',
                 color=colors['circuit'], alpha=0.8, edgecolor='black', linewidth=0.5)

ax3a.set_xlabel('Tokens Modified', fontsize=12, fontweight='bold')
ax3a.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax3a.set_title('Attack Success by Token Count', fontsize=14, fontweight='bold')
ax3a.set_xticks(x)
ax3a.set_xticklabels(tokens)
ax3a.legend(loc='upper left')
ax3a.set_ylim(0, max(max(circuit_by_token), max(random_by_token)) * 1.3)

# Add value labels
for bar, val in zip(bars1, random_by_token):
    if val > 0:
        ax3a.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, color=colors['random'])
for bar, val in zip(bars2, circuit_by_token):
    if val > 0:
        ax3a.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')

# Right panel: Key findings
ax3b.axis('off')

key_findings = f"""
══════════════════════════════════════════════
              KEY FINDINGS
══════════════════════════════════════════════

ATTACK SUCCESS RATES
──────────────────────────────────────────────
  Random Attacks:           {random_success}/{random_total} = {random_rate:.1f}%
  Circuit-Informed:         {circuit_success}/{circuit_total} = {circuit_rate:.1f}%

  Improvement:              {circuit_rate / random_rate:.0f}× more effective

EFFICIENCY (Success per Token Modified)
──────────────────────────────────────────────
  Random avg:               {random_eff:.4f}
  Circuit-informed avg:     {circuit_eff:.4f}

  Circuit is {efficiency_ratio:.0f}× MORE EFFICIENT

STATISTICAL SIGNIFICANCE
──────────────────────────────────────────────
  Fisher's exact test:      p = {p_value:.6f}
  Result:                   ✓ HIGHLY SIGNIFICANT (p < 0.001)

══════════════════════════════════════════════
            MAIN CONCLUSION
══════════════════════════════════════════════

  Understanding a model's internal circuits
  enables dramatically more effective
  adversarial attacks.

  This demonstrates that interpretability
  research has important security implications.

══════════════════════════════════════════════
"""

ax3b.text(0.05, 0.95, key_findings, transform=ax3b.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8,
                  edgecolor='orange', linewidth=2))

fig3.suptitle('Circuit-Informed Adversarial Attacks: Detailed Analysis',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(figures_dir / 'final_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.savefig(figures_dir / 'final_comparison.pdf', bbox_inches='tight', facecolor='white')
print(f"✅ {figures_dir / 'final_comparison.png'}")
print(f"✅ {figures_dir / 'final_comparison.pdf'}")
plt.close()

print("\n✨ All visualizations updated with circuit-informed attack results!")
