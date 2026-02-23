import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Add the sparse circuits package to path
SPARSE_CIRCUITS_PATH = Path(__file__).parent.parent / "research" / "sparse-circuits" / "circuit_sparsity-main"
sys.path.insert(0, str(SPARSE_CIRCUITS_PATH))

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR
from circuit_sparsity.tiktoken_ext import tinypython
from tiktoken import Encoding
from tiktoken.load import read_file_cached

def run_multi_seed_baseline():
    print("Loading resources for Multi-seed Baseline...")
    enc = Encoding(**tinypython.tinypython_2k())
    tok_double = enc.encode("]]\n")[0]
    tok_single = enc.encode("]\n")[0]
    
    model_name = "csp_yolo2"
    model_path = f"{MODEL_BASE_DIR}/models/{model_name}"
    model = load_model(model_path, flash=True, cuda=False)
    model.eval()

    viz_path = f"{MODEL_BASE_DIR}/viz/{model_name}/bracket_counting_beeg/prune_v4/k_optim/viz_data.pt"
    buf = io.BytesIO(read_file_cached(viz_path))
    viz_data = torch.load(buf, map_location="cpu", weights_only=False)
    samples_tensor = viz_data["importances"]["task_samples"][0]

    K = 10
    n_samples = 32
    budgets = [1, 2, 3, 5]
    
    print(f"Running Multi-seed Random Baseline (K={K}) for {n_samples} samples...")
    
    seeds = np.arange(42, 42 + K)
    all_success_rates = []

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(int(seed))
        
        seed_successes = 0
        seed_total = 0
        
        for i in range(n_samples):
            sample = samples_tensor[i]
            # Truncate zeros
            non_zero = sample.nonzero()
            if non_zero.numel() > 0:
                sample = sample[:non_zero[-1] + 1]
            
            with torch.no_grad():
                logits, _, _ = model(sample.unsqueeze(0))
                orig_pred = "]]" if logits[0, -1, tok_double] > logits[0, -1, tok_single] else "]"
            
            for b in budgets:
                attacked = sample.clone()
                if len(sample) > 5:
                    positions = np.random.choice(len(sample) - 5, b, replace=False)
                    for pos in positions:
                        attacked[pos] = np.random.randint(0, enc.n_vocab)
                
                with torch.no_grad():
                    logits_a, _, _ = model(attacked.unsqueeze(0))
                    attack_pred = "]]" if logits_a[0, -1, tok_double] > logits_a[0, -1, tok_single] else "]"
                
                if attack_pred != orig_pred:
                    seed_successes += 1
                seed_total += 1
                
        all_success_rates.append(seed_successes / seed_total)
        print(f"  Seed {seed} success rate: {all_success_rates[-1]*100:.2f}%")

    mean_sr = np.mean(all_success_rates)
    std_sr = np.std(all_success_rates, ddof=1)
    margin = 1.96 * (std_sr / np.sqrt(K))
    
    print(f"\nRandom Success Rate (K={K}): {mean_sr*100:.2f}% ± {margin*100:.2f}% (95% CI)")
    
    # Circuit attack success rate from previous results (approx 24.2%)
    circuit_sr = 0.242
    ratio = circuit_sr / mean_sr
    ratio_low = circuit_sr / (mean_sr + margin)
    ratio_high = circuit_sr / (mean_sr - margin) if mean_sr > margin else np.inf
    
    print(f"Efficiency Ratio: {ratio:.1f}x [95% CI: {ratio_low:.1f}x, {ratio_high:.1f}x]")

if __name__ == "__main__":
    run_multi_seed_baseline()
