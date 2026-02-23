"""
Gradient-Approximation Adversarial Attack Implementation
=========================================================

Uses fast sensitivity estimation to identify the most important token positions,
then modifies those positions to flip the model's prediction.

Optimized for speed by using a single probe token per position.
"""

import io
import sys
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
from tiktoken import Encoding
from tiktoken.load import read_file_cached

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "sparse-circuits" / "circuit_sparsity-main"))
from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


@dataclass
class AttackResult:
    sample_id: int
    attack_type: str
    attack_strategy: str
    tokens_modified: int
    original_prediction: str
    attacked_prediction: str
    prediction_flipped: bool
    original_logit_diff: float
    attacked_logit_diff: float
    efficiency: float


class GradientAttack:
    """
    Fast gradient-approximation adversarial attacks.

    Strategy: Probe each position with a bracket token to estimate sensitivity,
    then select the most sensitive positions for modification.
    """

    def __init__(self):
        self.enc = None
        self.model = None
        self.viz_data = None
        self.samples = []
        self.results = []

        self.tok_double = None
        self.tok_single = None
        self.tok_open_bracket = None

    def load(self):
        """Load model, tokenizer, and data"""
        print("Loading resources...", flush=True)

        self.enc = Encoding(**tinypython.tinypython_2k())
        self.tok_double = self.enc.encode("]]\n")[0]
        self.tok_single = self.enc.encode("]\n")[0]
        self.tok_open_bracket = self.enc.encode("[")[0]

        model_path = f"{MODEL_BASE_DIR}/models/csp_yolo2"
        self.model = load_model(model_path, flash=True, cuda=False)
        self.model.eval()

        viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
        buf = io.BytesIO(read_file_cached(viz_path))
        self.viz_data = torch.load(buf, map_location="cpu", weights_only=False)

        task_samples = self.viz_data["importances"]["task_samples"]
        samples_tensor = task_samples[0]

        for i in range(samples_tensor.shape[0]):
            sample = self._truncate_zeros(samples_tensor[i])
            self.samples.append({
                "id": i,
                "tokens": sample,
            })

        print(f"Loaded {len(self.samples)} samples", flush=True)
        return self

    def _truncate_zeros(self, sample):
        non_zero = sample.nonzero()
        if non_zero.numel() == 0:
            return sample
        return sample[:non_zero[-1] + 1]

    def _get_logit_diff(self, tokens):
        """Get logit difference (double - single)"""
        with torch.no_grad():
            logits, _, _ = self.model(tokens.unsqueeze(0))
            last = logits[0, -1]
            return float(last[self.tok_double] - last[self.tok_single])

    def _compute_fast_sensitivity(self, tokens, probe_positions=None):
        """
        Fast sensitivity computation: for each position, measure how much
        replacing with '[' changes the logit difference.

        Only probes positions where replacement differs from original.
        """
        seq_len = len(tokens)
        safe_len = seq_len - 5  # Don't modify near prediction

        if probe_positions is None:
            probe_positions = range(safe_len)

        baseline_diff = self._get_logit_diff(tokens)
        current_wants_decrease = baseline_diff > 0  # Currently predicting ]]

        sensitivities = {}

        for pos in probe_positions:
            if pos >= safe_len:
                continue
            if tokens[pos].item() == self.tok_open_bracket:
                continue  # Skip if already a bracket

            test_tokens = tokens.clone()
            test_tokens[pos] = self.tok_open_bracket
            new_diff = self._get_logit_diff(test_tokens)

            # Sensitivity = how much this helps us flip
            if current_wants_decrease:
                sensitivity = baseline_diff - new_diff  # Want to decrease diff
            else:
                sensitivity = new_diff - baseline_diff  # Want to increase diff

            sensitivities[pos] = sensitivity

        return sensitivities, baseline_diff

    def attack_sensitivity(self, tokens, n_modify):
        """
        Sensitivity-based attack:
        1. Compute sensitivity for each position
        2. Select top-n most sensitive positions
        3. Replace all with open brackets
        """
        attacked = tokens.clone()

        sensitivities, _ = self._compute_fast_sensitivity(tokens)

        if not sensitivities:
            return attacked

        # Sort by sensitivity (descending)
        sorted_positions = sorted(sensitivities.keys(),
                                  key=lambda p: sensitivities[p],
                                  reverse=True)

        # Modify top positions
        for pos in sorted_positions[:n_modify]:
            attacked[pos] = self.tok_open_bracket

        return attacked

    def attack_greedy(self, tokens, n_modify):
        """
        Greedy iterative attack:
        Apply modifications one at a time, recomputing sensitivity after each.
        """
        attacked = tokens.clone()
        modified = set()

        for _ in range(n_modify):
            # Get current prediction
            current_diff = self._get_logit_diff(attacked)
            current_pred = "]]" if current_diff > 0 else "]"

            # Compute sensitivities for unmodified positions
            remaining = [p for p in range(len(attacked) - 5) if p not in modified]
            sensitivities, _ = self._compute_fast_sensitivity(attacked, remaining)

            if not sensitivities:
                break

            # Find best position
            best_pos = max(sensitivities.keys(), key=lambda p: sensitivities[p])

            if sensitivities[best_pos] <= 0:
                break

            attacked[best_pos] = self.tok_open_bracket
            modified.add(best_pos)

            # Check if flipped
            new_diff = self._get_logit_diff(attacked)
            new_pred = "]]" if new_diff > 0 else "]"
            if new_pred != current_pred:
                break

        return attacked

    def run_all_attacks(self, tokens_to_modify=[1, 2, 3, 5]):
        """Run gradient attack strategies"""

        strategies = [
            ("gradient_sensitivity", self.attack_sensitivity),
            ("gradient_greedy", self.attack_greedy),
        ]

        print("\n" + "="*70, flush=True)
        print("RUNNING GRADIENT-BASED ATTACKS", flush=True)
        print("="*70, flush=True)

        for strategy_name, attack_fn in strategies:
            print(f"\n{'─'*70}", flush=True)
            print(f"Strategy: {strategy_name.upper()}", flush=True)
            print(f"{'─'*70}", flush=True)

            for n_tokens in tokens_to_modify:
                successes = 0
                total = 0

                for sample in self.samples:
                    tokens = sample["tokens"]

                    # Get original prediction
                    orig_diff = self._get_logit_diff(tokens)
                    orig_pred = "]]" if orig_diff > 0 else "]"
                    orig_ld = orig_diff if orig_diff > 0 else 0
                    orig_ls = -orig_diff if orig_diff < 0 else 0

                    # Apply attack
                    try:
                        attacked = attack_fn(tokens, n_tokens)
                    except Exception as e:
                        print(f"  Warning: {e}", flush=True)
                        attacked = tokens.clone()

                    # Get attacked prediction
                    attack_diff = self._get_logit_diff(attacked)
                    attack_pred = "]]" if attack_diff > 0 else "]"

                    flipped = (attack_pred != orig_pred)
                    if flipped:
                        successes += 1
                    total += 1

                    self.results.append(AttackResult(
                        sample_id=sample["id"],
                        attack_type="gradient",
                        attack_strategy=strategy_name,
                        tokens_modified=n_tokens,
                        original_prediction=orig_pred,
                        attacked_prediction=attack_pred,
                        prediction_flipped=flipped,
                        original_logit_diff=orig_diff,
                        attacked_logit_diff=attack_diff,
                        efficiency=(1.0 / n_tokens) if flipped else 0.0
                    ))

                success_rate = successes / total * 100
                print(f"  {n_tokens} token(s): {successes}/{total} = {success_rate:.1f}%", flush=True)

        return self

    def save_results(self, filename="gradient_attack_results.csv"):
        """Save results to CSV"""
        output_path = Path(__file__).parent.parent / 'data' / filename

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'sample_id', 'attack_type', 'attack_strategy', 'tokens_modified',
                'original_prediction', 'attacked_prediction', 'prediction_flipped',
                'original_logit_diff', 'attacked_logit_diff', 'efficiency'
            ])
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))

        print(f"\n💾 Saved to {output_path}", flush=True)
        return output_path

    def analyze(self):
        """Analyze gradient attack results"""
        print("\n" + "="*70, flush=True)
        print("ANALYSIS: GRADIENT ATTACK STRATEGIES", flush=True)
        print("="*70, flush=True)

        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in self.results])

        summary = df.groupby(['attack_strategy', 'tokens_modified']).agg({
            'prediction_flipped': ['sum', 'count', 'mean'],
            'efficiency': 'mean'
        }).reset_index()
        summary.columns = ['strategy', 'tokens', 'successes', 'total', 'success_rate', 'avg_efficiency']

        print("\nBy Strategy and Token Count:", flush=True)
        print(summary.to_string(index=False), flush=True)

        print("\n" + "-"*70, flush=True)
        overall = df.groupby('attack_strategy').agg({
            'prediction_flipped': 'mean',
            'efficiency': 'mean'
        }).reset_index()
        overall.columns = ['strategy', 'overall_success_rate', 'overall_efficiency']
        overall = overall.sort_values('overall_success_rate', ascending=False)

        print("OVERALL RANKING:", flush=True)
        for i, row in overall.iterrows():
            print(f"  {row['strategy']}: {row['overall_success_rate']*100:.1f}% success", flush=True)

        return summary


if __name__ == "__main__":
    print("="*70, flush=True)
    print("GRADIENT-BASED ADVERSARIAL ATTACK EXPERIMENT", flush=True)
    print("="*70, flush=True)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    attack = GradientAttack()
    attack.load()
    attack.run_all_attacks([1, 2, 3, 5])
    attack.save_results()
    attack.analyze()

    print("\n" + "="*70, flush=True)
    print("EXPERIMENT COMPLETE", flush=True)
    print("="*70, flush=True)
