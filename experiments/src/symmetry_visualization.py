"""
Symmetry Visualization: Geometric Interpretation of the Directional Collision Principle
========================================================================================

This script creates a visualization showing how attacks move samples toward
the decision boundary (the τ-symmetry axis where g(v) = 0).

KEY CONCEPT:
- g(v) = A(v) - A(τ(v)) = logit_diff
- Decision boundary is where g(v) = 0
- Attacks succeed by pushing samples TOWARD this boundary
- Samples on opposite sides need opposite push directions

The visualization shows:
1. Samples positioned by their logit_diff (g(v) value)
2. Decision boundary at g(v) = 0
3. Attack vectors showing direction and magnitude of shift
4. Color coding by attack success
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

EXPERIMENTS_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENTS_DIR / 'data'
FIGURES_DIR = EXPERIMENTS_DIR / 'figures'


def load_data():
    """Load and prepare data for visualization"""
    df = pd.read_csv(DATA_DIR / 'circuit_informed_results.csv')
    df['logit_shift'] = df['attacked_logit_diff'] - df['original_logit_diff']
    return df


def create_1d_symmetry_figure(df, output_path):
    """
    Create 1D visualization showing samples on the g(v) axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Get per-sample data
    by_sample = df.groupby('sample_id').agg({
        'original_logit_diff': 'first',
        'original_prediction': 'first',
        'prediction_flipped': 'max',  # Ever flipped?
    }).reset_index()

    by_sample['vulnerable'] = by_sample['original_logit_diff'] < 0

    # =========================================================================
    # Panel A: Samples on g(v) axis with decision boundary
    # =========================================================================
    ax = axes[0, 0]

    # Plot samples
    robust = by_sample[~by_sample['vulnerable']]
    vulnerable = by_sample[by_sample['vulnerable']]

    ax.scatter(robust['original_logit_diff'], [1] * len(robust),
               s=200, c='steelblue', alpha=0.7, edgecolors='black',
               label=f'Robust (g(v) > 0): {len(robust)} samples', zorder=5)
    ax.scatter(vulnerable['original_logit_diff'], [1] * len(vulnerable),
               s=200, c='coral', alpha=0.7, edgecolors='black',
               label=f'Vulnerable (g(v) < 0): {len(vulnerable)} samples', zorder=5)

    # Decision boundary
    ax.axvline(0, color='red', linewidth=3, linestyle='-', label='Decision boundary: g(v) = 0')

    # Shade regions
    xlim = max(abs(by_sample['original_logit_diff'].min()), by_sample['original_logit_diff'].max()) * 1.1
    ax.axvspan(-xlim, 0, alpha=0.1, color='coral', label='Vulnerable region')
    ax.axvspan(0, xlim, alpha=0.1, color='steelblue', label='Robust region')

    # Labels
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0.5, 1.5)
    ax.set_yticks([])
    ax.set_xlabel('g(v) = logit("]]") - logit("]")', fontsize=12)
    ax.set_title('A. Samples on the Symmetry Axis\ng(v) = 0 is the decision boundary',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)

    # Add arrows showing attack direction
    ax.annotate('', xy=(-2, 1.3), xytext=(-8, 1.3),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    ax.text(-5, 1.35, 'Attacks push\ntoward boundary', ha='center', fontsize=10, color='darkgreen')

    ax.annotate('', xy=(2, 1.3), xytext=(8, 1.3),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax.text(5, 1.35, 'Would need to push\naway (wrong direction)', ha='center', fontsize=10, color='darkred')

    # =========================================================================
    # Panel B: Attack vectors showing shift direction
    # =========================================================================
    ax = axes[0, 1]

    # Get attack data for combined strategy (best performer)
    combined = df[df['attack_strategy'] == 'combined'].copy()
    combined_by_sample = combined.groupby('sample_id').agg({
        'original_logit_diff': 'first',
        'logit_shift': 'mean',
        'prediction_flipped': 'max',
    }).reset_index()

    # Position samples vertically by original logit_diff
    y_positions = np.linspace(0.1, 0.9, len(combined_by_sample))
    combined_by_sample['y'] = y_positions

    # Plot arrows
    for _, row in combined_by_sample.iterrows():
        x_start = row['original_logit_diff']
        x_end = x_start + row['logit_shift']
        y = row['y']

        color = '#27ae60' if row['prediction_flipped'] else '#c0392b'
        alpha = 0.8 if row['prediction_flipped'] else 0.4

        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=alpha))

        # Mark original position
        ax.scatter([x_start], [y], s=100, c='black', zorder=5)

    # Decision boundary
    ax.axvline(0, color='red', linewidth=3, linestyle='-')
    ax.axvspan(-xlim, 0, alpha=0.1, color='coral')
    ax.axvspan(0, xlim, alpha=0.1, color='steelblue')

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('g(v) = logit_diff', fontsize=12)
    ax.set_title('B. Attack Vectors (Combined Strategy)\nGreen = flipped, Red = not flipped',
                 fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#27ae60', alpha=0.8, label='Successful attack'),
        mpatches.Patch(color='#c0392b', alpha=0.4, label='Failed attack'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Decision boundary'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # =========================================================================
    # Panel C: Before/After positions for flipped samples
    # =========================================================================
    ax = axes[1, 0]

    # Get samples that flipped
    flipped = df[(df['attack_strategy'] == 'combined') & (df['prediction_flipped'] == True)]
    flipped_samples = flipped.groupby('sample_id').agg({
        'original_logit_diff': 'first',
        'attacked_logit_diff': 'mean',
    }).reset_index()

    if len(flipped_samples) > 0:
        y_pos = np.linspace(0.2, 0.8, len(flipped_samples))

        for i, (_, row) in enumerate(flipped_samples.iterrows()):
            # Before
            ax.scatter([row['original_logit_diff']], [y_pos[i]], s=150, c='coral',
                       edgecolors='black', marker='o', zorder=5)
            # After
            ax.scatter([row['attacked_logit_diff']], [y_pos[i]], s=150, c='steelblue',
                       edgecolors='black', marker='s', zorder=5)
            # Arrow
            ax.annotate('', xy=(row['attacked_logit_diff'], y_pos[i]),
                        xytext=(row['original_logit_diff'], y_pos[i]),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.axvline(0, color='red', linewidth=3, linestyle='-')
    ax.axvspan(-xlim, 0, alpha=0.1, color='coral')
    ax.axvspan(0, xlim, alpha=0.1, color='steelblue')

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('g(v) = logit_diff', fontsize=12)
    ax.set_title('C. Successful Attacks Cross the Boundary\nCircle = before, Square = after',
                 fontsize=14, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='coral', markersize=10,
                   linestyle='None', label='Before attack'),
        plt.Line2D([0], [0], marker='s', color='steelblue', markersize=10,
                   linestyle='None', label='After attack'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # =========================================================================
    # Panel D: Summary
    # =========================================================================
    ax = axes[1, 1]
    ax.axis('off')

    summary = """
    ══════════════════════════════════════════════════════════════
            GEOMETRIC INTERPRETATION: τ-SYMMETRY AXIS
    ══════════════════════════════════════════════════════════════

    THE KEY GEOMETRIC INSIGHT
    ─────────────────────────────────────────────────────────────

    The Directional Collision Principle has a simple geometric
    interpretation:

    • g(v) = A(v) - A(τ(v)) = logit_diff
    • Decision boundary is where g(v) = 0
    • τ-involution swaps the two classes (]] ↔ ])

    WHAT THE VISUALIZATION SHOWS
    ─────────────────────────────────────────────────────────────

    Panel A: Samples positioned by their g(v) value
             • Vulnerable samples (g < 0) on the left
             • Robust samples (g > 0) on the right

    Panel B: Attack vectors showing shift direction
             • Green arrows → successful attacks (crossed boundary)
             • Red arrows → failed attacks (didn't reach boundary)

    Panel C: Before/After for flipped samples
             • All successful attacks CROSSED the g(v) = 0 line

    WHY DIRECTION MATTERS
    ─────────────────────────────────────────────────────────────

    For g(v) < 0 samples (vulnerable):
      → Need POSITIVE shift to reach boundary
      → Circuit attacks achieve this efficiently

    For g(v) > 0 samples (robust):
      → Would need NEGATIVE shift
      → Our attacks push in wrong direction → 0% success

    This explains the PERFECT SELECTIVITY: attacks only succeed
    on samples where they push TOWARD the boundary.

    ══════════════════════════════════════════════════════════════
    """

    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan',
                     edgecolor='steelblue', alpha=0.9))

    fig.suptitle('Symmetry Visualization: The Geometry of Adversarial Attacks',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    print("=" * 70)
    print("SYMMETRY VISUALIZATION")
    print("Geometric interpretation of the Directional Collision Principle")
    print("=" * 70)

    # Load data
    print("\n[1/2] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} attack trials")

    # Create figure
    print("\n[2/2] Creating symmetry visualization...")
    output_path = FIGURES_DIR / 'symmetry_visualization.png'
    create_1d_symmetry_figure(df, output_path)
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.with_suffix('.pdf')}")

    # Summary stats
    by_sample = df.groupby('sample_id').agg({
        'original_logit_diff': 'first',
        'prediction_flipped': 'max',
    }).reset_index()

    vulnerable = by_sample[by_sample['original_logit_diff'] < 0]
    robust = by_sample[by_sample['original_logit_diff'] > 0]

    print(f"\n{'='*70}")
    print("GEOMETRIC SUMMARY")
    print("="*70)
    print(f"""
    Total samples: {len(by_sample)}

    Vulnerable region (g(v) < 0): {len(vulnerable)} samples
      - Ever flipped: {vulnerable['prediction_flipped'].sum()}
      - Attack success rate: {vulnerable['prediction_flipped'].mean()*100:.1f}%

    Robust region (g(v) > 0): {len(robust)} samples
      - Ever flipped: {robust['prediction_flipped'].sum()}
      - Attack success rate: {robust['prediction_flipped'].mean()*100:.1f}%

    DIRECTIONAL SELECTIVITY:
    Attacks succeed ONLY on vulnerable samples (g < 0) where
    they push TOWARD the decision boundary.

    This is the geometric meaning of the Directional Collision Principle.
    """)

    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
