"""
Certified Lipschitz Bound Computation for Sparse Transformers
=============================================================

Computes a certified upper bound on the Lipschitz constant of the
csp_yolo2 (bracket) and csp_yolo1 (quote) sparse transformer models
by composing spectral norms of weight matrices through the architecture.

Key formula for each transformer block (residual stream):
    Lip(block) <= 1 + Lip(attn_branch) + Lip(mlp_branch)

where:
    Lip(attn) <= sigma_1(c_attn) * sigma_1(c_proj)
    Lip(mlp)  <= sigma_1(c_fc) * Lip(act_fn) * sigma_1(c_proj)

Full network:
    G_total = sigma_1(wte) * prod_{i=0}^{n_layer-1} Lip(block_i) * sigma_1(lm_head)

The certified bound G_total is compared to the empirical L-hat values.
"""

import json
import math
import sys
from pathlib import Path

import torch
import numpy as np

# Add the circuit_sparsity package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "research" / "sparse-circuits" / "circuit_sparsity-main"))

from circuit_sparsity.inference.gpt import load_model
from circuit_sparsity.registries import MODEL_BASE_DIR


# GELU Lipschitz constant: max |GELU'(x)| approx 1.7
GELU_LIP = 1.7
RELU_LIP = 1.0


def spectral_norm(weight: torch.Tensor) -> float:
    """Compute the largest singular value of a weight matrix."""
    W = weight.detach().float()
    if W.is_sparse:
        W = W.to_dense()
    if W.ndim == 1:
        return float(W.abs().max())
    return float(torch.linalg.svdvals(W)[0])


def sparsity_fraction(weight: torch.Tensor) -> float:
    """Fraction of weights that are exactly zero."""
    W = weight.detach()
    if W.is_sparse:
        total = 1
        for s in W.shape:
            total *= s
        return 1.0 - W._nnz() / total
    return float((W == 0).sum()) / W.numel()


def compute_layer_bounds(model) -> list[dict]:
    """Compute per-layer spectral norms and Lipschitz bounds."""
    config = model.config
    act_lip = GELU_LIP if config.activation_type == "gelu" else RELU_LIP

    layer_data = []
    for i, block in enumerate(model.transformer.h):
        attn = block.attn
        mlp = block.mlp

        # Attention branch: sigma(c_attn) * sigma(c_proj)
        sigma_c_attn = spectral_norm(attn.c_attn.weight)
        sigma_c_proj_attn = spectral_norm(attn.c_proj.weight)
        lip_attn = sigma_c_attn * sigma_c_proj_attn

        # MLP branch: sigma(c_fc) * act_lip * sigma(c_proj)
        sigma_c_fc = spectral_norm(mlp.c_fc.weight)
        sigma_c_proj_mlp = spectral_norm(mlp.c_proj.weight)
        lip_mlp = sigma_c_fc * act_lip * sigma_c_proj_mlp

        # Block with residual: 1 + attn + mlp
        lip_block = 1.0 + lip_attn + lip_mlp

        # Sparsity stats
        sp_c_attn = sparsity_fraction(attn.c_attn.weight)
        sp_c_proj_attn = sparsity_fraction(attn.c_proj.weight)
        sp_c_fc = sparsity_fraction(mlp.c_fc.weight)
        sp_c_proj_mlp = sparsity_fraction(mlp.c_proj.weight)

        layer_data.append({
            "layer": i,
            "sigma_c_attn": sigma_c_attn,
            "sigma_c_proj_attn": sigma_c_proj_attn,
            "lip_attn": lip_attn,
            "sp_c_attn": sp_c_attn,
            "sp_c_proj_attn": sp_c_proj_attn,
            "sigma_c_fc": sigma_c_fc,
            "sigma_c_proj_mlp": sigma_c_proj_mlp,
            "lip_mlp": lip_mlp,
            "sp_c_fc": sp_c_fc,
            "sp_c_proj_mlp": sp_c_proj_mlp,
            "lip_block": lip_block,
            "mean_sparsity": (sp_c_attn + sp_c_proj_attn + sp_c_fc + sp_c_proj_mlp) / 4,
        })

    return layer_data


def compute_certified_bound(model) -> dict:
    """Compute the full certified Lipschitz bound for the network."""
    # Embedding spectral norm
    sigma_wte = spectral_norm(model.transformer.wte.weight)
    sp_wte = sparsity_fraction(model.transformer.wte.weight)

    # LM head spectral norm (may be tied to wte)
    sigma_lm_head = spectral_norm(model.lm_head.weight)
    sp_lm_head = sparsity_fraction(model.lm_head.weight)

    # Per-layer bounds
    layer_data = compute_layer_bounds(model)

    # Product of block Lipschitz constants
    log_product = sum(math.log(ld["lip_block"]) for ld in layer_data)

    # G_total = sigma(wte) * product(lip_block_i) * sigma(lm_head)
    log_G_total = math.log(sigma_wte) + log_product + math.log(sigma_lm_head)
    G_total = math.exp(log_G_total)

    # For two logit outputs from same network: G1 + G2 <= 2 * G_total
    G_certified = 2.0 * G_total

    return {
        "sigma_wte": sigma_wte,
        "sp_wte": sp_wte,
        "sigma_lm_head": sigma_lm_head,
        "sp_lm_head": sp_lm_head,
        "layer_data": layer_data,
        "log_product_blocks": log_product,
        "G_total": G_total,
        "log_G_total": log_G_total,
        "G_certified": G_certified,
        "log_G_certified": math.log(G_certified),
    }


def format_results(name: str, result: dict, L_hat: float) -> str:
    """Format results as a readable report."""
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"  CERTIFIED LIPSCHITZ BOUND: {name}")
    lines.append(f"{'='*70}")

    lines.append(f"\n  Embedding:  sigma(wte) = {result['sigma_wte']:.4f}  "
                 f"(sparsity {result['sp_wte']:.1%})")
    lines.append(f"  LM Head:    sigma(lm)  = {result['sigma_lm_head']:.4f}  "
                 f"(sparsity {result['sp_lm_head']:.1%})")

    lines.append(f"\n  {'Layer':>5} | {'sigma(c_attn)':>13} | {'sigma(c_proj)':>13} | "
                 f"{'Lip(attn)':>10} | {'sigma(c_fc)':>11} | {'sigma(c_proj)':>13} | "
                 f"{'Lip(mlp)':>10} | {'Lip(block)':>10} | {'Sparsity':>8}")
    lines.append(f"  {'-'*5}-+-{'-'*13}-+-{'-'*13}-+-{'-'*10}-+-{'-'*11}-+-"
                 f"{'-'*13}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for ld in result["layer_data"]:
        lines.append(
            f"  {ld['layer']:>5} | {ld['sigma_c_attn']:>13.4f} | {ld['sigma_c_proj_attn']:>13.4f} | "
            f"{ld['lip_attn']:>10.2f} | {ld['sigma_c_fc']:>11.4f} | {ld['sigma_c_proj_mlp']:>13.4f} | "
            f"{ld['lip_mlp']:>10.2f} | {ld['lip_block']:>10.2f} | {ld['mean_sparsity']:>7.1%}"
        )

    lines.append(f"\n  Composition (log scale):")
    lines.append(f"    log(sigma_wte)       = {math.log(result['sigma_wte']):>10.4f}")
    lines.append(f"    sum log(Lip(block))  = {result['log_product_blocks']:>10.4f}")
    lines.append(f"    log(sigma_lm_head)   = {math.log(result['sigma_lm_head']):>10.4f}")
    lines.append(f"    log(G_total)         = {result['log_G_total']:>10.4f}")
    lines.append(f"    log(G_certified)     = {result['log_G_certified']:>10.4f}")

    lines.append(f"\n  CERTIFIED BOUND:")
    lines.append(f"    G_total     = {result['G_total']:.6e}")
    lines.append(f"    G_certified = 2 * G_total = {result['G_certified']:.6e}")

    lines.append(f"\n  COMPARISON TO EMPIRICAL:")
    lines.append(f"    L-hat (calibrated)  = {L_hat:.4f}")
    lines.append(f"    G_certified         = {result['G_certified']:.6e}")
    ratio = result['G_certified'] / L_hat
    lines.append(f"    Tightness gap       = {ratio:.2e}x  (log10 = {math.log10(ratio):.1f})")

    lines.append(f"\n  CERTIFIED EXCLUSION RADIUS (for |m(x)| = 1.0):")
    certified_radius = 1.0 / result['G_certified']
    calibrated_radius = 1.0 / L_hat
    lines.append(f"    Certified:  {certified_radius:.6e} tokens")
    lines.append(f"    Calibrated: {calibrated_radius:.4f} tokens")
    lines.append(f"    Ratio:      {calibrated_radius / certified_radius:.2e}x tighter")

    return "\n".join(lines)


def generate_figure(results: dict, output_dir: Path):
    """Generate comparison figure with per-layer spectral norms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [matplotlib not available, skipping figure generation]")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Certified Lipschitz Bound: Per-Layer Analysis", fontsize=14, fontweight="bold")

    for idx, (name, res) in enumerate(results.items()):
        layer_data = res["result"]["layer_data"]
        layers = [ld["layer"] for ld in layer_data]

        # Top row: spectral norms
        ax = axes[0][idx]
        width = 0.35
        x = np.arange(len(layers))
        bars1 = ax.bar(x - width/2, [ld["lip_attn"] for ld in layer_data],
                       width, label="Lip(attn)", color="#4472C4", alpha=0.85)
        bars2 = ax.bar(x + width/2, [ld["lip_mlp"] for ld in layer_data],
                       width, label="Lip(mlp)", color="#ED7D31", alpha=0.85)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Branch Lipschitz Bound")
        ax.set_title(f"{name}")
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend(fontsize=8)
        ax.set_yscale("log")

        # Bottom row: sparsity
        ax2 = axes[1][idx]
        sparsities_attn = [(ld["sp_c_attn"] + ld["sp_c_proj_attn"]) / 2 * 100 for ld in layer_data]
        sparsities_mlp = [(ld["sp_c_fc"] + ld["sp_c_proj_mlp"]) / 2 * 100 for ld in layer_data]
        ax2.bar(x - width/2, sparsities_attn, width, label="Attn sparsity", color="#4472C4", alpha=0.6)
        ax2.bar(x + width/2, sparsities_mlp, width, label="MLP sparsity", color="#ED7D31", alpha=0.6)
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Weight Sparsity (%)")
        ax2.set_title(f"{name} — Sparsity")
        ax2.set_xticks(x)
        ax2.set_xticklabels(layers)
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 100)

    plt.tight_layout()
    fig_path = output_dir / "certified_lipschitz_bounds.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")
    return fig_path


def main():
    experiments_dir = Path(__file__).parent.parent
    data_dir = experiments_dir / 'data'
    figures_dir = experiments_dir / 'figures'

    params_path = data_dir / "estimated_parameters.json"
    with open(params_path) as f:
        params = json.load(f)

    L_hat_bracket = params["bracket_task"]["lipschitz_constant"]
    L_hat_quote = params["quote_task"]["lipschitz_constant"]

    # Models to analyze
    models_config = {
        "csp_yolo2 (bracket)": {
            "path": f"{MODEL_BASE_DIR}/models/csp_yolo2",
            "L_hat": L_hat_bracket,
        },
        "csp_yolo1 (quote)": {
            "path": f"{MODEL_BASE_DIR}/models/csp_yolo1",
            "L_hat": L_hat_quote,
        },
    }

    all_results = {}

    for name, cfg in models_config.items():
        print(f"\n{'#'*70}")
        print(f"  Loading {name}...")
        print(f"{'#'*70}")

        model = load_model(cfg["path"], flash=True, cuda=False)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        n_zero = sum(int((p.detach() == 0).sum()) if not p.is_sparse
                     else (p.numel() - p._nnz())
                     for p in model.parameters())
        print(f"  Parameters: {n_params:,}  |  Zeros: {n_zero:,}  |  "
              f"Overall sparsity: {n_zero/n_params:.1%}")

        result = compute_certified_bound(model)
        print(format_results(name, result, cfg["L_hat"]))

        all_results[name] = {"result": result, "L_hat": cfg["L_hat"]}

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary comparison table
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"\n  {'Model':<25} | {'L-hat':>10} | {'G_certified':>15} | {'Gap (log10)':>12} | {'Mean Sparsity':>14}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*15}-+-{'-'*12}-+-{'-'*14}")

    certified_values = {}
    for name, data in all_results.items():
        res = data["result"]
        L_hat = data["L_hat"]
        ratio = res["G_certified"] / L_hat
        mean_sp = np.mean([ld["mean_sparsity"] for ld in res["layer_data"]])
        print(f"  {name:<25} | {L_hat:>10.4f} | {res['G_certified']:>15.4e} | "
              f"{math.log10(ratio):>12.1f} | {mean_sp:>13.1%}")
        certified_values[name] = {
            "G_total": res["G_total"],
            "G_certified": res["G_certified"],
            "log10_gap": math.log10(ratio),
        }

    # Generate figure
    generate_figure(all_results, figures_dir)

    # Update estimated_parameters.json with certified bounds
    for name, data in all_results.items():
        res = data["result"]
        task_key = "bracket_task" if "bracket" in name else "quote_task"
        params[task_key]["certified_G_total"] = res["G_total"]
        params[task_key]["certified_G_bound"] = res["G_certified"]
        params[task_key]["certified_log10_gap"] = math.log10(res["G_certified"] / data["L_hat"])
        params[task_key]["model_overall_sparsity"] = float(
            np.mean([ld["mean_sparsity"] for ld in res["layer_data"]])
        )

    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\n  Updated: {params_path}")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
