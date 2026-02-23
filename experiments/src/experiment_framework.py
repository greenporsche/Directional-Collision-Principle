"""
Circuit-Targeted Adversarial Attack Experiment Framework
========================================================

Research Question:
How does the attack efficiency (success rate per token modified) of circuit-targeted
adversarial attacks on the bracket_counting circuit compare to random token attacks?

Model: GPT-2 csp_yolo2 (475M parameters)
Circuit: bracket_counting
Samples: 32 pre-generated Python code snippets
Task: Predict ']]' vs ']' bracket closing

Metrics:
- Primary: Attack efficiency = success_rate / tokens_modified
- Secondary: Raw success rate
"""

import io
import csv
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Literal

import torch
import numpy as np
from tiktoken import Encoding
from tiktoken.load import read_file_cached

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "research" / "sparse-circuits" / "circuit_sparsity-main"))

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ExperimentResult:
    """Single experiment trial result"""
    sample_id: int
    attack_type: Literal["clean", "random", "circuit_targeted", "gradient"]
    tokens_modified: int
    original_prediction: str  # "]]" or "]"
    attacked_prediction: str  # "]]" or "]"
    prediction_flipped: bool
    original_logit_diff: float  # logit(correct) - logit(incorrect)
    attacked_logit_diff: float
    efficiency: float  # For individual trials: (1 if flipped else 0) / tokens_modified


@dataclass
class SampleData:
    """Pre-loaded sample data"""
    sample_id: int
    tokens: torch.Tensor
    decoded_text: str
    correct_answer: str  # "]]" or "]" - what the model SHOULD predict
    seq_length: int


# =============================================================================
# Experiment Manager
# =============================================================================

class BracketCountingExperiment:
    """
    Manages the bracket counting adversarial attack experiment.

    Usage:
        exp = BracketCountingExperiment()
        exp.load_data()
        exp.run_clean_baseline()
        exp.run_random_attacks(tokens_to_modify=[1, 2, 3, 5])
        exp.run_circuit_attacks(tokens_to_modify=[1, 2, 3, 5])
        exp.save_results()
        exp.analyze()
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or Path(__file__).parent.parent / 'data')
        self.results: list[ExperimentResult] = []
        self.samples: list[SampleData] = []

        # Token IDs for bracket predictions
        self.tok_double = None  # ']]\\n'
        self.tok_single = None  # ']\\n'

        # Model and tokenizer
        self.model = None
        self.enc = None

        # Viz data with activations
        self.viz_data = None

        print("Experiment framework initialized")
        print(f"Output directory: {self.output_dir}")

    def load_data(self):
        """Load model, tokenizer, and samples"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)

        # Initialize tokenizer
        print("\n[1/3] Loading tokenizer...")
        self.enc = Encoding(**tinypython.tinypython_2k())
        self.tok_double = self.enc.encode("]]\n")[0]
        self.tok_single = self.enc.encode("]\n")[0]
        print(f"  Token for ']]\\n': {self.tok_double}")
        print(f"  Token for ']\\n': {self.tok_single}")

        # Load model
        print("\n[2/3] Loading model (csp_yolo2)...")
        model_path = f"{MODEL_BASE_DIR}/models/csp_yolo2"
        self.model = load_model(model_path, flash=True, cuda=False)
        self.model.eval()
        print(f"  Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")

        # Load viz data with samples
        print("\n[3/3] Loading bracket_counting samples...")
        viz_path = f"{MODEL_BASE_DIR}/viz/csp_yolo2/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
        buf = io.BytesIO(read_file_cached(viz_path))
        self.viz_data = torch.load(buf, map_location="cpu", weights_only=False)

        # Extract samples
        task_samples = self.viz_data["importances"]["task_samples"]
        samples_tensor = task_samples[0]  # [32, 237]

        print(f"  Raw samples shape: {samples_tensor.shape}")

        # Process each sample
        for i in range(samples_tensor.shape[0]):
            sample = self._truncate_zeros(samples_tensor[i])
            decoded = self.enc.decode(sample.tolist())

            # Determine correct answer by analyzing the code structure
            # (In practice, we'll determine this from model's clean prediction)
            self.samples.append(SampleData(
                sample_id=i,
                tokens=sample,
                decoded_text=decoded,
                correct_answer="TBD",  # Will be set by baseline
                seq_length=len(sample)
            ))

        print(f"  Loaded {len(self.samples)} samples")
        print(f"  Average sequence length: {np.mean([s.seq_length for s in self.samples]):.1f} tokens")

        print("\n✅ Data loading complete!")
        return self

    def _truncate_zeros(self, sample: torch.Tensor) -> torch.Tensor:
        """Remove trailing zero padding from sample"""
        non_zero = sample.nonzero()
        if non_zero.numel() == 0:
            return sample
        return sample[:non_zero[-1] + 1]

    def _get_prediction(self, tokens: torch.Tensor) -> tuple[str, float, float]:
        """
        Run model on tokens and get prediction.

        Returns:
            (predicted_token, logit_double, logit_single)
        """
        with torch.no_grad():
            logits, _, _ = self.model(tokens.unsqueeze(0))
            last_logits = logits[0, -1]

            logit_double = float(last_logits[self.tok_double])
            logit_single = float(last_logits[self.tok_single])

            if logit_double > logit_single:
                return "]]", logit_double, logit_single
            else:
                return "]", logit_double, logit_single

    def run_clean_baseline(self) -> dict:
        """
        Run all samples through clean model to establish baseline.
        Sets the 'correct_answer' for each sample based on model's prediction.

        Returns dict with baseline statistics.
        """
        print("\n" + "="*60)
        print("RUNNING CLEAN BASELINE")
        print("="*60)

        predictions = {"]]": 0, "]": 0}

        for sample in self.samples:
            pred, logit_double, logit_single = self._get_prediction(sample.tokens)

            # Set the correct answer as what the clean model predicts
            sample.correct_answer = pred
            predictions[pred] += 1

            # Record baseline result
            self.results.append(ExperimentResult(
                sample_id=sample.sample_id,
                attack_type="clean",
                tokens_modified=0,
                original_prediction=pred,
                attacked_prediction=pred,
                prediction_flipped=False,
                original_logit_diff=logit_double - logit_single,
                attacked_logit_diff=logit_double - logit_single,
                efficiency=0.0  # N/A for baseline
            ))

            if (sample.sample_id + 1) % 8 == 0:
                print(f"  Processed {sample.sample_id + 1}/32 samples...")

        print(f"\n📊 Baseline Results:")
        print(f"  Predictions of ']]': {predictions[']]']}")
        print(f"  Predictions of ']': {predictions[']']}")
        print(f"  Total samples: {len(self.samples)}")

        return {
            "total_samples": len(self.samples),
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }

    def run_random_attacks(self, tokens_to_modify: list[int] = [1, 2, 3, 5]) -> dict:
        """
        Run random token substitution attacks.

        For each sample and each modification count:
        - Randomly select N token positions (not the last few which contain the answer)
        - Replace with random tokens from vocabulary
        - Measure if prediction flips

        Args:
            tokens_to_modify: List of token counts to test [1, 2, 3, 5]
        """
        print("\n" + "="*60)
        print("RUNNING RANDOM ATTACKS")
        print("="*60)

        vocab_size = self.model.config.vocab_size
        results_summary = {n: {"flipped": 0, "total": 0} for n in tokens_to_modify}

        for n_tokens in tokens_to_modify:
            print(f"\n  Testing with {n_tokens} token(s) modified...")

            for sample in self.samples:
                # Get original prediction
                orig_pred, orig_logit_d, orig_logit_s = self._get_prediction(sample.tokens)

                # Create attacked version
                attacked_tokens = sample.tokens.clone()

                # Select random positions (avoid last 5 tokens which may be critical)
                safe_positions = list(range(max(0, len(attacked_tokens) - 10)))
                if len(safe_positions) < n_tokens:
                    safe_positions = list(range(len(attacked_tokens) - 1))

                positions = np.random.choice(safe_positions, size=min(n_tokens, len(safe_positions)), replace=False)

                # Replace with random tokens
                for pos in positions:
                    attacked_tokens[pos] = np.random.randint(0, vocab_size)

                # Get attacked prediction
                attack_pred, attack_logit_d, attack_logit_s = self._get_prediction(attacked_tokens)

                flipped = (attack_pred != orig_pred)
                if flipped:
                    results_summary[n_tokens]["flipped"] += 1
                results_summary[n_tokens]["total"] += 1

                self.results.append(ExperimentResult(
                    sample_id=sample.sample_id,
                    attack_type="random",
                    tokens_modified=n_tokens,
                    original_prediction=orig_pred,
                    attacked_prediction=attack_pred,
                    prediction_flipped=flipped,
                    original_logit_diff=orig_logit_d - orig_logit_s,
                    attacked_logit_diff=attack_logit_d - attack_logit_s,
                    efficiency=(1.0 / n_tokens) if flipped else 0.0
                ))

            success_rate = results_summary[n_tokens]["flipped"] / results_summary[n_tokens]["total"]
            print(f"    Success rate: {success_rate:.1%} ({results_summary[n_tokens]['flipped']}/{results_summary[n_tokens]['total']})")

        return results_summary

    def run_circuit_attacks(self, tokens_to_modify: list[int] = [1, 2, 3, 5]) -> dict:
        """
        Run circuit-targeted attacks using interpretability knowledge.

        Strategy: Use knowledge of bracket-counting circuit to craft attacks:
        1. Add distractor brackets in "safe" positions (comments, strings)
        2. Insert unmatched brackets that confuse the counting
        3. Modify tokens that activate the bracket-detection neurons

        The full circuit-targeted implementation is in circuit_informed_attack.py;
        this method uses a heuristic approach based on bracket semantics.
        """
        print("\n" + "="*60)
        print("RUNNING CIRCUIT-TARGETED ATTACKS")
        print("="*60)

        # Heuristic approach based on bracket semantics
        # Full circuit-targeted implementation is in circuit_informed_attack.py

        results_summary = {n: {"flipped": 0, "total": 0} for n in tokens_to_modify}

        # Get bracket-related tokens
        bracket_tokens = {
            "open": self.enc.encode("[")[0],
            "close": self.enc.encode("]")[0],
            "open_paren": self.enc.encode("(")[0],
            "close_paren": self.enc.encode(")")[0],
        }

        for n_tokens in tokens_to_modify:
            print(f"\n  Testing with {n_tokens} token(s) modified (circuit-targeted)...")

            for sample in self.samples:
                # Get original prediction
                orig_pred, orig_logit_d, orig_logit_s = self._get_prediction(sample.tokens)

                # Create attacked version using circuit knowledge
                attacked_tokens = sample.tokens.clone()

                # Strategy: Insert/modify brackets to confuse the counter
                # Find positions near existing brackets and modify strategically
                token_list = attacked_tokens.tolist()

                # Find bracket positions
                bracket_positions = []
                for i, tok in enumerate(token_list[:-5]):  # Avoid last 5
                    decoded = self.enc.decode([tok])
                    if '[' in decoded or ']' in decoded:
                        bracket_positions.append(i)

                # If we found brackets, target nearby positions
                if bracket_positions and n_tokens > 0:
                    # Insert extra opening brackets to throw off the count
                    target_positions = []
                    for bp in bracket_positions[:n_tokens]:
                        # Try position just before a bracket
                        if bp > 0:
                            target_positions.append(bp - 1)

                    # Fill remaining with positions near brackets
                    while len(target_positions) < n_tokens and bracket_positions:
                        bp = bracket_positions[len(target_positions) % len(bracket_positions)]
                        if bp + 1 < len(token_list) - 5:
                            target_positions.append(bp + 1)
                        else:
                            target_positions.append(max(0, bp - 2))

                    # Apply modifications - insert bracket-related tokens
                    for i, pos in enumerate(target_positions[:n_tokens]):
                        if i % 2 == 0:
                            attacked_tokens[pos] = bracket_tokens["open"]
                        else:
                            attacked_tokens[pos] = bracket_tokens["close"]
                else:
                    # Fallback: modify random positions with brackets
                    positions = np.random.choice(
                        range(max(1, len(attacked_tokens) - 10)),
                        size=min(n_tokens, len(attacked_tokens) - 10),
                        replace=False
                    )
                    for pos in positions:
                        attacked_tokens[pos] = bracket_tokens["open"]

                # Get attacked prediction
                attack_pred, attack_logit_d, attack_logit_s = self._get_prediction(attacked_tokens)

                flipped = (attack_pred != orig_pred)
                if flipped:
                    results_summary[n_tokens]["flipped"] += 1
                results_summary[n_tokens]["total"] += 1

                self.results.append(ExperimentResult(
                    sample_id=sample.sample_id,
                    attack_type="circuit_targeted",
                    tokens_modified=n_tokens,
                    original_prediction=orig_pred,
                    attacked_prediction=attack_pred,
                    prediction_flipped=flipped,
                    original_logit_diff=orig_logit_d - orig_logit_s,
                    attacked_logit_diff=attack_logit_d - attack_logit_s,
                    efficiency=(1.0 / n_tokens) if flipped else 0.0
                ))

            success_rate = results_summary[n_tokens]["flipped"] / results_summary[n_tokens]["total"]
            print(f"    Success rate: {success_rate:.1%} ({results_summary[n_tokens]['flipped']}/{results_summary[n_tokens]['total']})")

        return results_summary

    def save_results(self, filename: str = "experiment_results.csv"):
        """Save all results to CSV"""
        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'sample_id', 'attack_type', 'tokens_modified',
                'original_prediction', 'attacked_prediction', 'prediction_flipped',
                'original_logit_diff', 'attacked_logit_diff', 'efficiency'
            ])
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))

        print(f"\n💾 Results saved to: {filepath}")
        return filepath

    def analyze(self) -> dict:
        """
        Analyze results and compute efficiency metrics.

        Returns summary statistics for each attack type.
        """
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)

        # Group results by attack type and tokens modified
        from collections import defaultdict
        grouped = defaultdict(lambda: defaultdict(list))

        for r in self.results:
            if r.attack_type != "clean":
                grouped[r.attack_type][r.tokens_modified].append(r)

        analysis = {}

        for attack_type in ["random", "circuit_targeted"]:
            if attack_type not in grouped:
                continue

            print(f"\n📊 {attack_type.upper()} ATTACKS:")
            analysis[attack_type] = {}

            for n_tokens in sorted(grouped[attack_type].keys()):
                results = grouped[attack_type][n_tokens]
                success_rate = sum(1 for r in results if r.prediction_flipped) / len(results)
                efficiency = success_rate / n_tokens if n_tokens > 0 else 0

                analysis[attack_type][n_tokens] = {
                    "success_rate": success_rate,
                    "efficiency": efficiency,
                    "n_samples": len(results)
                }

                print(f"  {n_tokens} token(s): Success={success_rate:.1%}, Efficiency={efficiency:.3f}")

        # Compare overall
        print("\n" + "-"*40)
        print("COMPARISON SUMMARY:")

        for n_tokens in [1, 2, 3, 5]:
            if "random" in analysis and "circuit_targeted" in analysis:
                rand_eff = analysis.get("random", {}).get(n_tokens, {}).get("efficiency", 0)
                circ_eff = analysis.get("circuit_targeted", {}).get(n_tokens, {}).get("efficiency", 0)

                if rand_eff > 0:
                    improvement = (circ_eff - rand_eff) / rand_eff * 100
                    print(f"  {n_tokens} tokens: Circuit is {improvement:+.1f}% {'better' if improvement > 0 else 'worse'} than random")

        return analysis


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("BRACKET COUNTING ADVERSARIAL ATTACK EXPERIMENT")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize and run experiment
    exp = BracketCountingExperiment()
    exp.load_data()

    # Run baseline
    baseline_stats = exp.run_clean_baseline()

    # Save intermediate results
    exp.save_results("baseline_results.csv")

    print("\n" + "="*60)
    print("BASELINE COMPLETE - Ready for attack experiments")
    print("="*60)
    print("\nNext steps:")
    print("  exp.run_random_attacks([1, 2, 3, 5])")
    print("  exp.run_circuit_attacks([1, 2, 3, 5])")
    print("  exp.save_results()")
    print("  exp.analyze()")
