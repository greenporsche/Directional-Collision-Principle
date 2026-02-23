"""
Circuit-Informed Adversarial Attack Implementation
=================================================

Based on the OpenAI Circuit Sparsity paper, this implements attacks that:
1. Use knowledge of which neurons matter for bracket counting
2. Target specific positions that feed into those neurons
3. Use distractor patterns identified in the paper (Figure 8)

Key Circuit Neurons for Bracket Counting:
- Layer 0 MLP: channels [2719, 4016, 4607, 5116, 5553, 6416, 7275]
- Layer 7 MLP: channels [1482, 2199, 2288, 2511, 3082, 4133, 6250, ...]
"""

import io
import sys
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal

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


class CircuitInformedAttack:
    """
    Implements circuit-informed attacks based on OpenAI paper insights.
    """

    def __init__(self):
        self.enc = None
        self.model = None
        self.viz_data = None
        self.samples = []
        self.results = []

        # Token IDs
        self.tok_double = None  # ']]'
        self.tok_single = None  # ']'

        # Key circuit information (from analysis)
        self.circuit_neurons = {
            "0.mlp.post_act": [2719, 4016, 4607, 5116, 5553, 6416, 7275],
            "7.mlp.post_act": [1482, 2199, 2288, 2511, 3082, 4133, 6250, 6323, 6561, 6720, 7058, 7618],
        }

        # Important token patterns from paper
        self.attack_tokens = {}

    def load(self):
        """Load model, tokenizer, and data"""
        print("Loading resources...")

        # Tokenizer
        self.enc = Encoding(**tinypython.tinypython_2k())
        self.tok_double = self.enc.encode("]]\n")[0]
        self.tok_single = self.enc.encode("]\n")[0]

        # Important tokens for attacks
        self.attack_tokens = {
            "open_bracket": self.enc.encode("[")[0],
            "close_bracket": self.enc.encode("]")[0],
            "hash": self.enc.encode("#")[0],
            "quote": self.enc.encode('"')[0],
            "space": self.enc.encode(" ")[0],
            "comma": self.enc.encode(",")[0],
            "newline": self.enc.encode("\n")[0],
        }

        # Model
        model_path = f"{MODEL_BASE_DIR}/models/csp_yolo2"
        self.model = load_model(model_path, flash=True, cuda=False)
        self.model.eval()

        # Viz data with samples and activations
        viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
        buf = io.BytesIO(read_file_cached(viz_path))
        self.viz_data = torch.load(buf, map_location="cpu", weights_only=False)

        # Load samples
        task_samples = self.viz_data["importances"]["task_samples"]
        samples_tensor = task_samples[0]
        self.activations = task_samples[1]

        for i in range(samples_tensor.shape[0]):
            sample = self._truncate_zeros(samples_tensor[i])
            self.samples.append({
                "id": i,
                "tokens": sample,
                "text": self.enc.decode(sample.tolist()),
            })

        print(f"Loaded {len(self.samples)} samples")
        return self

    def _truncate_zeros(self, sample):
        non_zero = sample.nonzero()
        if non_zero.numel() == 0:
            return sample
        return sample[:non_zero[-1] + 1]

    def _get_prediction(self, tokens):
        with torch.no_grad():
            logits, _, _ = self.model(tokens.unsqueeze(0))
            last = logits[0, -1]
            ld = float(last[self.tok_double])
            ls = float(last[self.tok_single])
            pred = "]]" if ld > ls else "]"
            return pred, ld, ls

    def _find_bracket_positions(self, tokens):
        """Find positions of bracket-related tokens"""
        positions = {"open": [], "close": [], "all": []}
        for i, tok in enumerate(tokens.tolist()):
            decoded = self.enc.decode([tok])
            if "[" in decoded:
                positions["open"].append(i)
                positions["all"].append(i)
            if "]" in decoded:
                positions["close"].append(i)
                positions["all"].append(i)
        return positions

    def _find_high_activation_positions(self, sample_id, layer="0.mlp.post_act", top_k=10):
        """
        Find token positions where key circuit neurons have highest activation.
        These are the positions most "important" to the circuit's computation.
        """
        layer_acts = self.activations.get(layer, {})
        if not layer_acts:
            return []

        # Average activation across all key channels
        all_activations = []
        for ch in self.circuit_neurons.get(layer, []):
            if ch in layer_acts:
                ch_act = layer_acts[ch]
                if isinstance(ch_act, torch.Tensor) and ch_act.dim() >= 2:
                    all_activations.append(ch_act[sample_id].abs())

        if not all_activations:
            return []

        # Sum activations across channels
        combined = torch.stack(all_activations).sum(dim=0)

        # Get top positions (excluding last 5 tokens)
        safe_len = len(combined) - 5
        if safe_len <= 0:
            return []

        top_positions = torch.topk(combined[:safe_len], min(top_k, safe_len)).indices.tolist()
        return top_positions

    # =========================================================================
    # ATTACK STRATEGIES (Based on OpenAI Paper)
    # =========================================================================

    def attack_distractor_comment(self, tokens, n_modify):
        """
        Strategy 1: Add distractor brackets in a comment
        From paper: "distractor brackets in comments confuse the counter"

        Insert: # [[[  at a strategic position
        """
        attacked = tokens.clone()
        token_list = attacked.tolist()

        # Find a good position (after a newline, or near brackets)
        bracket_pos = self._find_bracket_positions(attacked)

        # Build distractor: # [[[
        distractor = [
            self.attack_tokens["hash"],
            self.attack_tokens["space"],
        ] + [self.attack_tokens["open_bracket"]] * min(n_modify, 5)

        # Insert position: just before the main bracket expression
        if bracket_pos["open"]:
            insert_pos = max(0, bracket_pos["open"][0] - 2)
        else:
            insert_pos = len(token_list) // 2

        # Insert distractor (replace tokens at position)
        for i, tok in enumerate(distractor[:n_modify]):
            if insert_pos + i < len(attacked) - 5:
                attacked[insert_pos + i] = tok

        return attacked

    def attack_unbalanced_context(self, tokens, n_modify):
        """
        Strategy 2: Add unbalanced brackets earlier in context
        From paper: "unmatched open brackets before the target confuse counting"

        Insert unmatched [ tokens in the first half of the sequence
        """
        attacked = tokens.clone()
        seq_len = len(attacked)

        # Target first third of sequence
        target_region = seq_len // 3

        # Find positions that aren't already brackets
        safe_positions = []
        for i in range(min(target_region, seq_len - 10)):
            decoded = self.enc.decode([attacked[i].item()])
            if "[" not in decoded and "]" not in decoded:
                safe_positions.append(i)

        # Insert unbalanced open brackets
        if safe_positions:
            positions = np.random.choice(
                safe_positions,
                size=min(n_modify, len(safe_positions)),
                replace=False
            )
            for pos in positions:
                attacked[pos] = self.attack_tokens["open_bracket"]

        return attacked

    def attack_activation_targeted(self, tokens, n_modify, sample_id):
        """
        Strategy 3: Target positions with highest circuit activation
        Modify tokens at positions where the bracket-counting neurons fire most
        """
        attacked = tokens.clone()

        # Find high-activation positions
        high_act_positions = self._find_high_activation_positions(sample_id)

        if not high_act_positions:
            # Fallback to bracket positions
            bracket_pos = self._find_bracket_positions(attacked)
            high_act_positions = bracket_pos["all"][:10]

        # Modify those positions with confusing tokens
        positions_to_modify = high_act_positions[:n_modify]

        for i, pos in enumerate(positions_to_modify):
            if pos < len(attacked) - 5:
                # Alternate between open and close brackets to confuse counter
                if i % 2 == 0:
                    attacked[pos] = self.attack_tokens["open_bracket"]
                else:
                    attacked[pos] = self.attack_tokens["close_bracket"]

        return attacked

    def attack_context_dilution(self, tokens, n_modify):
        """
        Strategy 4: Dilute the bracket signal with many tokens
        From paper: "long lists dilute the context and reduce bracket tracking accuracy"

        Insert many comma-separated values before brackets
        """
        attacked = tokens.clone()
        bracket_pos = self._find_bracket_positions(attacked)

        if not bracket_pos["open"]:
            return attacked

        # Insert pattern: 1, 2, 3, 4, ... before the bracket
        insert_pos = max(0, bracket_pos["open"][0] - n_modify)

        # Create dilution pattern (numbers and commas)
        dilution_tokens = []
        for i in range(n_modify):
            # Add digit tokens
            digit = str(i % 10)
            try:
                dilution_tokens.append(self.enc.encode(digit)[0])
            except:
                dilution_tokens.append(self.attack_tokens["comma"])

        # Insert dilution
        for i, tok in enumerate(dilution_tokens):
            if insert_pos + i < len(attacked) - 5:
                attacked[insert_pos + i] = tok

        return attacked

    def attack_combined(self, tokens, n_modify, sample_id):
        """
        Strategy 5: Combined attack using multiple strategies
        Uses both activation targeting AND distractor insertion
        """
        attacked = tokens.clone()

        # Half modifications: activation targeting
        n_activation = n_modify // 2
        high_act_positions = self._find_high_activation_positions(sample_id)[:n_activation]

        for i, pos in enumerate(high_act_positions):
            if pos < len(attacked) - 5:
                attacked[pos] = self.attack_tokens["open_bracket"]

        # Half modifications: add distractors near brackets
        n_distractor = n_modify - n_activation
        bracket_pos = self._find_bracket_positions(tokens)

        if bracket_pos["open"] and n_distractor > 0:
            insert_pos = max(0, bracket_pos["open"][0] - n_distractor)
            for i in range(n_distractor):
                if insert_pos + i < len(attacked) - 5:
                    attacked[insert_pos + i] = self.attack_tokens["open_bracket"]

        return attacked

    # =========================================================================
    # RUN EXPERIMENTS
    # =========================================================================

    def run_all_attacks(self, tokens_to_modify=[1, 2, 3, 5]):
        """Run all attack strategies"""

        strategies = [
            ("distractor_comment", self.attack_distractor_comment),
            ("unbalanced_context", self.attack_unbalanced_context),
            ("activation_targeted", lambda t, n, sid=None: self.attack_activation_targeted(t, n, sid)),
            ("context_dilution", self.attack_context_dilution),
            ("combined", lambda t, n, sid=None: self.attack_combined(t, n, sid)),
        ]

        print("\n" + "="*70)
        print("RUNNING CIRCUIT-INFORMED ATTACKS")
        print("="*70)

        for strategy_name, attack_fn in strategies:
            print(f"\n{'─'*70}")
            print(f"Strategy: {strategy_name.upper()}")
            print(f"{'─'*70}")

            for n_tokens in tokens_to_modify:
                successes = 0
                total = 0

                for sample in self.samples:
                    # Get original prediction
                    orig_pred, orig_ld, orig_ls = self._get_prediction(sample["tokens"])

                    # Apply attack
                    if "targeted" in strategy_name or "combined" in strategy_name:
                        attacked = attack_fn(sample["tokens"], n_tokens, sample["id"])
                    else:
                        attacked = attack_fn(sample["tokens"], n_tokens)

                    # Get attacked prediction
                    attack_pred, attack_ld, attack_ls = self._get_prediction(attacked)

                    flipped = (attack_pred != orig_pred)
                    if flipped:
                        successes += 1
                    total += 1

                    self.results.append(AttackResult(
                        sample_id=sample["id"],
                        attack_type="circuit_informed",
                        attack_strategy=strategy_name,
                        tokens_modified=n_tokens,
                        original_prediction=orig_pred,
                        attacked_prediction=attack_pred,
                        prediction_flipped=flipped,
                        original_logit_diff=orig_ld - orig_ls,
                        attacked_logit_diff=attack_ld - attack_ls,
                        efficiency=(1.0 / n_tokens) if flipped else 0.0
                    ))

                success_rate = successes / total * 100
                print(f"  {n_tokens} token(s): {successes}/{total} = {success_rate:.1f}%")

        return self

    def save_results(self, filename="circuit_informed_results.csv"):
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

        print(f"\n💾 Saved to {output_path}")
        return output_path

    def analyze(self):
        """Analyze and compare strategies"""
        print("\n" + "="*70)
        print("ANALYSIS: CIRCUIT-INFORMED ATTACK STRATEGIES")
        print("="*70)

        import pandas as pd
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Summary by strategy
        summary = df.groupby(['attack_strategy', 'tokens_modified']).agg({
            'prediction_flipped': ['sum', 'count', 'mean'],
            'efficiency': 'mean'
        }).reset_index()
        summary.columns = ['strategy', 'tokens', 'successes', 'total', 'success_rate', 'avg_efficiency']

        print("\nBy Strategy and Token Count:")
        print(summary.to_string(index=False))

        # Best strategy per token count
        print("\n" + "-"*70)
        print("BEST STRATEGY BY TOKEN COUNT:")
        for n in [1, 2, 3, 5]:
            subset = summary[summary['tokens'] == n]
            if len(subset) > 0:
                best = subset.loc[subset['success_rate'].idxmax()]
                print(f"  {n} token(s): {best['strategy']} ({best['success_rate']*100:.1f}% success)")

        # Overall best
        print("\n" + "-"*70)
        overall = df.groupby('attack_strategy').agg({
            'prediction_flipped': 'mean',
            'efficiency': 'mean'
        }).reset_index()
        overall.columns = ['strategy', 'overall_success_rate', 'overall_efficiency']
        overall = overall.sort_values('overall_success_rate', ascending=False)

        print("OVERALL RANKING:")
        for i, row in overall.iterrows():
            print(f"  {row['strategy']}: {row['overall_success_rate']*100:.1f}% success, {row['overall_efficiency']:.4f} efficiency")

        return summary


if __name__ == "__main__":
    print("="*70)
    print("CIRCUIT-INFORMED ADVERSARIAL ATTACK EXPERIMENT")
    print("Based on OpenAI Circuit Sparsity Paper Insights")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    attack = CircuitInformedAttack()
    attack.load()
    attack.run_all_attacks([1, 2, 3, 5])
    attack.save_results()
    attack.analyze()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
