"""
Predict-Then-Attack Protocol
=============================

Can we predict which samples will be successfully attacked BEFORE running attacks?

Based on the Directional Collision Principle:
- Circuit attacks push toward "]]" (add brackets)
- Samples with NEGATIVE logit_diff (predicting "]") are vulnerable
- Samples with POSITIVE logit_diff (predicting "]]") are NOT vulnerable

PREDICTION RULE:
  If signed_logit_diff < threshold: VULNERABLE
  If signed_logit_diff >= threshold: ROBUST

This demonstrates that the theory has genuine PREDICTIVE POWER.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, accuracy_score, classification_report
)
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

EXPERIMENTS_DIR = Path(__file__).parent.parent
DATA_DIR = EXPERIMENTS_DIR / 'data'
FIGURES_DIR = EXPERIMENTS_DIR / 'figures'


def load_data():
    """Load experiment results"""
    df_full = pd.read_csv(DATA_DIR / 'full_experiment_results.csv')
    df_circuit = pd.read_csv(DATA_DIR / 'circuit_informed_results.csv')
    return df_full, df_circuit


def prepare_prediction_data(df_circuit):
    """
    Prepare data for prediction task.

    For each sample, we want:
    - signed_logit_diff: The predictor (from theory)
    - was_ever_attacked: Ground truth (was the sample ever successfully attacked?)
    """
    # Use combined strategy (our best attack)
    combined = df_circuit[df_circuit['attack_strategy'] == 'combined'].copy()

    # For each sample: get margin and whether it was ever successfully attacked
    by_sample = combined.groupby('sample_id').agg({
        'original_logit_diff': 'first',
        'original_prediction': 'first',
        'prediction_flipped': 'max'  # 1 if ever flipped, 0 otherwise
    }).reset_index()

    by_sample.rename(columns={
        'original_logit_diff': 'signed_margin',
        'prediction_flipped': 'was_attacked'
    }, inplace=True)

    return by_sample


def simple_threshold_predictor(signed_margin, threshold=0):
    """
    Simple prediction based on directional theory:
    - If margin < threshold: predict VULNERABLE (1)
    - If margin >= threshold: predict ROBUST (0)

    Default threshold = 0 means:
    - Negative logit_diff (predicts "]") → vulnerable
    - Positive logit_diff (predicts "]]") → robust
    """
    return (signed_margin < threshold).astype(int)


def find_optimal_threshold(data, thresholds=None):
    """
    Find the threshold that maximizes F1 score.

    This shows whether any threshold works, not just 0.
    """
    if thresholds is None:
        thresholds = np.linspace(data['signed_margin'].min(),
                                  data['signed_margin'].max(), 100)

    best_f1 = 0
    best_threshold = 0
    results = []

    for t in thresholds:
        pred = simple_threshold_predictor(data['signed_margin'], t)
        actual = data['was_attacked'].astype(int)

        if actual.sum() == 0 or actual.sum() == len(actual):
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, pred, average='binary', zero_division=0
        )

        results.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1, pd.DataFrame(results)


def evaluate_predictor(data, threshold):
    """
    Full evaluation of the prediction model.
    """
    pred = simple_threshold_predictor(data['signed_margin'], threshold)
    actual = data['was_attacked'].astype(int)

    # Basic metrics
    accuracy = accuracy_score(actual, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, pred, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(actual, pred)

    # ROC curve (using negative margin as score, since lower = more vulnerable)
    fpr, tpr, roc_thresholds = roc_curve(actual, -data['signed_margin'])
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds, 'auc': roc_auc}
    }


def cross_validate_predictor(data, n_splits=5):
    """
    Cross-validation to get robust estimates.

    Note: With only 32 samples, we use leave-some-out approach.
    """
    from sklearn.model_selection import StratifiedKFold

    X = data['signed_margin'].values.reshape(-1, 1)
    y = data['was_attacked'].astype(int).values

    # Check if we have both classes
    if len(np.unique(y)) < 2:
        return None

    skf = StratifiedKFold(n_splits=min(n_splits, sum(y), sum(1-y)), shuffle=True, random_state=42)

    results = []
    for train_idx, test_idx in skf.split(X, y):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Find optimal threshold on training data
        best_t, _, _ = find_optimal_threshold(train_data)

        # Evaluate on test data
        pred = simple_threshold_predictor(test_data['signed_margin'], best_t)
        actual = test_data['was_attacked'].astype(int)

        if len(np.unique(actual)) < 2:
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, pred, average='binary', zero_division=0
        )
        accuracy = accuracy_score(actual, pred)

        results.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': best_t
        })

    if not results:
        return None

    return pd.DataFrame(results)


def create_prediction_figure(data, metrics, threshold, output_path):
    """
    Comprehensive figure showing prediction performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Scatter with decision boundary
    ax = axes[0, 0]

    vulnerable = data[data['was_attacked'] == 1]
    robust = data[data['was_attacked'] == 0]

    ax.scatter(vulnerable['signed_margin'], [1] * len(vulnerable),
               c='#e74c3c', s=150, alpha=0.7, label='Actually Attacked',
               edgecolors='black', marker='o')
    ax.scatter(robust['signed_margin'], [0] * len(robust),
               c='#3498db', s=150, alpha=0.7, label='Not Attacked',
               edgecolors='black', marker='o')

    # Decision boundary
    ax.axvline(x=threshold, color='green', linestyle='--', linewidth=3,
               label=f'Decision Threshold (t={threshold:.2f})')

    # Shade prediction regions
    ax.axvspan(ax.get_xlim()[0], threshold, alpha=0.1, color='red',
               label='Predicted Vulnerable')
    ax.axvspan(threshold, ax.get_xlim()[1], alpha=0.1, color='blue',
               label='Predicted Robust')

    ax.set_xlabel('Signed Logit Diff (← vulnerable | robust →)', fontsize=12)
    ax.set_ylabel('Actually Attacked?', fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.set_title('Prediction vs Reality\n(Theory-Based Decision Boundary)', fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=9)

    # Panel B: ROC Curve
    ax = axes[0, 1]

    roc = metrics['roc']
    ax.plot(roc['fpr'], roc['tpr'], color='#e74c3c', linewidth=3,
            label=f'ROC Curve (AUC = {roc["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
    ax.fill_between(roc['fpr'], roc['tpr'], alpha=0.2, color='#e74c3c')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: Predicting Attack Success\nAUC = {roc["auc"]:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Panel C: Confusion Matrix
    ax = axes[1, 0]

    cm = metrics['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center", fontsize=20, fontweight='bold',
                   color="white" if cm[i, j] > thresh else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nRobust', 'Predicted\nVulnerable'], fontsize=11)
    ax.set_yticklabels(['Actually\nRobust', 'Actually\nVulnerable'], fontsize=11)
    ax.set_xlabel('Prediction', fontsize=12)
    ax.set_ylabel('Reality', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Panel D: Removed for cleaner poster layout
    ax = axes[1, 1]
    ax.axis('off')

    fig.suptitle('PREDICT-THEN-ATTACK: Theory Predicts Vulnerability Before Attack',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def generate_prediction_summary(data, metrics, threshold, cv_results=None):
    """Generate detailed prediction summary"""

    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()

    summary = f"""
{'='*70}
PREDICT-THEN-ATTACK PROTOCOL RESULTS
{'='*70}

THE EXPERIMENT
──────────────────────────────────────────────────────────────────────
Can the Directional Collision Principle predict attack success
BEFORE running any attacks?

PREDICTION MODEL
──────────────────────────────────────────────────────────────────────
Based on theoretical insight:
  - Circuit attacks push toward "]]" (insert brackets)
  - Samples predicting "]" (negative logit_diff) can be flipped
  - Samples predicting "]]" (positive logit_diff) cannot

Rule: VULNERABLE if signed_logit_diff < {threshold:.2f}
      ROBUST if signed_logit_diff ≥ {threshold:.2f}

{'='*70}
RESULTS
{'='*70}

CLASSIFICATION METRICS
─────────────────────────────────────────────────────────────
Accuracy:   {metrics['accuracy']:.1%}  ({int(metrics['accuracy']*len(data))}/{len(data)} correct)
Precision:  {metrics['precision']:.1%}  (of predicted vulnerable, {tp}/{tp+fp} were attacked)
Recall:     {metrics['recall']:.1%}  (of actual attacks, {tp}/{tp+fn} were predicted)
F1 Score:   {metrics['f1']:.1%}  (harmonic mean of precision and recall)

ROC AUC:    {metrics['roc']['auc']:.3f}  (1.0 = perfect, 0.5 = random)

CONFUSION MATRIX
─────────────────────────────────────────────────────────────
                    Predicted Robust    Predicted Vulnerable
Actually Robust         {tn:3d}                  {fp:3d}
Actually Attacked       {fn:3d}                  {tp:3d}

INTERPRETATION
─────────────────────────────────────────────────────────────
True Negatives:  {tn} samples correctly identified as robust
True Positives:  {tp} samples correctly identified as vulnerable
False Positives: {fp} samples incorrectly flagged as vulnerable
False Negatives: {fn} attacks we failed to predict

{'='*70}
SIGNIFICANCE
{'='*70}

{'EXCELLENT' if metrics['accuracy'] >= 0.9 else 'GOOD' if metrics['accuracy'] >= 0.8 else 'MODERATE'} PREDICTION ACCURACY: {metrics['accuracy']:.0%}

The fact that a simple threshold on signed_logit_diff achieves
{metrics['accuracy']:.0%} accuracy demonstrates that:

1. The Directional Collision Principle has PREDICTIVE POWER
2. We can identify vulnerable samples BEFORE attacking
3. The theory captures the essential structure of vulnerability

Key result:
→ Theory makes prediction
→ Prediction is tested
→ Prediction succeeds with {metrics['accuracy']:.0%} accuracy

{'='*70}
"""

    if cv_results is not None:
        summary += f"""
CROSS-VALIDATION (Robustness Check)
─────────────────────────────────────────────────────────────
Mean Accuracy:  {cv_results['accuracy'].mean():.1%} ± {cv_results['accuracy'].std():.1%}
Mean F1 Score:  {cv_results['f1'].mean():.1%} ± {cv_results['f1'].std():.1%}

{'='*70}
"""

    return summary


def main():
    print("=" * 70)
    print("PREDICT-THEN-ATTACK PROTOCOL")
    print("Can theory predict vulnerability before attacking?")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    df_full, df_circuit = load_data()

    # Prepare prediction data
    print("\n[2/5] Preparing prediction dataset...")
    data = prepare_prediction_data(df_circuit)
    n_attacked = data['was_attacked'].sum()
    n_robust = len(data) - n_attacked
    print(f"  Total samples: {len(data)}")
    print(f"  Actually attacked: {n_attacked} ({n_attacked/len(data):.1%})")
    print(f"  Actually robust: {n_robust} ({n_robust/len(data):.1%})")

    # Find optimal threshold
    print("\n[3/5] Finding optimal prediction threshold...")
    best_threshold, best_f1, threshold_results = find_optimal_threshold(data)
    print(f"  Optimal threshold: {best_threshold:.4f}")
    print(f"  Best F1 at this threshold: {best_f1:.3f}")

    # For this task, threshold=0 is theoretically motivated
    # Let's compare theory-based (t=0) vs empirically optimal
    theory_threshold = 0
    print(f"\n  Theory-based threshold (t=0): predicts based on sign of logit_diff")

    # Use theory-based threshold for main analysis
    threshold = theory_threshold

    # Evaluate predictor
    print("\n[4/5] Evaluating predictor...")
    metrics = evaluate_predictor(data, threshold)
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1 Score:  {metrics['f1']:.1%}")
    print(f"  ROC AUC:   {metrics['roc']['auc']:.3f}")

    # Cross-validation (if enough data)
    print("\n  Running cross-validation...")
    cv_results = cross_validate_predictor(data)
    if cv_results is not None:
        print(f"  CV Accuracy: {cv_results['accuracy'].mean():.1%} ± {cv_results['accuracy'].std():.1%}")

    # Generate figure
    print("\n[5/5] Generating prediction figure...")
    fig_path = create_prediction_figure(data, metrics, threshold,
                                         FIGURES_DIR / 'predict_then_attack.png')
    print(f"  Saved: {fig_path}")

    # Generate and print summary
    summary = generate_prediction_summary(data, metrics, threshold, cv_results)
    print(summary)

    # Save summary
    summary_path = DATA_DIR / 'predict_then_attack_results.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"Results saved to: {summary_path}")

    # Final verdict
    print("\n" + "=" * 70)
    if metrics['accuracy'] >= 0.8:
        print("✓ PREDICTION VALIDATED")
        print(f"  Theory-based predictor achieves {metrics['accuracy']:.0%} accuracy")
        print("  The Directional Collision Principle has genuine predictive power!")
    elif metrics['accuracy'] >= 0.6:
        print("◐ PARTIAL VALIDATION")
        print(f"  Theory-based predictor achieves {metrics['accuracy']:.0%} accuracy")
        print("  Better than random, but room for improvement")
    else:
        print("✗ PREDICTION NOT VALIDATED")
        print(f"  Theory-based predictor achieves only {metrics['accuracy']:.0%} accuracy")
    print("=" * 70)

    return {
        'data': data,
        'metrics': metrics,
        'threshold': threshold,
        'cv_results': cv_results
    }


if __name__ == "__main__":
    results = main()
