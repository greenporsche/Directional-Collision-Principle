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

def run_icp():
    print("Loading resources for ICP...")
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

    M = 10
    n_samples = 32
    results = []
    
    # FP IDs identified: [18, 21, 28, 29]
    fp_ids = [18, 21, 28, 29]

    print(f"Running ICP with M={M} for {n_samples} samples...")
    
    np.random.seed(42)
    torch.manual_seed(42)

    for i in range(n_samples):
        sample = samples_tensor[i]
        # Truncate zeros
        non_zero = sample.nonzero()
        if non_zero.numel() > 0:
            sample = sample[:non_zero[-1] + 1]
        
        # Original logit diff
        with torch.no_grad():
            logits, _, _ = model(sample.unsqueeze(0))
            last = logits[0, -1]
            orig_ld = float(last[tok_double])
            orig_ls = float(last[tok_single])
            orig_g = orig_ld - orig_ls
        
        shifts = []
        for _ in range(M):
            # Pick a random position (excluding the last few)
            if len(sample) <= 5: continue
            pos = np.random.randint(0, len(sample) - 5)
            # Pick a random token from vocab
            new_tok = np.random.randint(0, enc.n_vocab)
            
            attacked = sample.clone()
            attacked[pos] = new_tok
            
            with torch.no_grad():
                logits_a, _, _ = model(attacked.unsqueeze(0))
                last_a = logits_a[0, -1]
                ld_a = float(last_a[tok_double])
                ls_a = float(last_a[tok_single])
                g_a = ld_a - ls_a
            
            shifts.append(abs(g_a - orig_g))
            
        l_local_icp = np.mean(shifts) if shifts else 0
        is_fp = i in fp_ids
        is_vulnerable_pred = orig_g < 0
        
        results.append({
            'sample_id': i,
            'orig_g': orig_g,
            'l_local_icp': l_local_icp,
            'is_fp': is_fp,
            'is_vulnerable_pred': is_vulnerable_pred
        })
        if i % 1 == 0: 
            print(f"  Processed {i}/{n_samples}...", flush=True)

    df = pd.DataFrame(results)
    
    print("\nICP Results Summary:", flush=True)
    tp_group = df[(df['is_vulnerable_pred']) & (~df['is_fp'])]
    fp_group = df[df['is_fp']]
    
    print(f"Mean L_local_ICP for True Positives (n={len(tp_group)}): {tp_group['l_local_icp'].mean():.4f}")
    print(f"Mean L_local_ICP for False Positives (n={len(fp_group)}): {fp_group['l_local_icp'].mean():.4f}")
    
    ratio = tp_group['l_local_icp'].mean() / fp_group['l_local_icp'].mean()
    print(f"Ratio TP/FP L_local_ICP: {ratio:.2f}x")
    
    data_dir = Path(__file__).parent.parent / 'data'
    df.to_csv(data_dir / 'icp_results.csv', index=False)
    print(f"Saved results to {data_dir / 'icp_results.csv'}")

if __name__ == "__main__":
    run_icp()
