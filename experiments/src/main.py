"""
Directional Collision Principle — Experiment Runner
====================================================

Runs the core validation experiments in sequence.
Each script can also be run independently from experiments/:

    python src/directional_analysis.py
    python src/predict_then_attack.py
    python src/attack_taxonomy_analysis.py
    python src/lipschitz_estimation.py
    python src/quote_lipschitz_estimation.py
    python src/quote_bracket_comparison.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "directional_analysis.py",
    "predict_then_attack.py",
    "attack_taxonomy_analysis.py",
    "lipschitz_estimation.py",
    "quote_lipschitz_estimation.py",
    "quote_task_validation.py",
    "quote_bracket_comparison.py",
    "error_analysis.py",
    "defense_demonstration.py",
]

def main():
    src_dir = Path(__file__).parent
    failed = []

    for script in SCRIPTS:
        path = src_dir / script
        print(f"\n{'='*70}")
        print(f"Running: {script}")
        print(f"{'='*70}")
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            failed.append(script)
            print(f"  ** {script} exited with code {result.returncode}")

    print(f"\n{'='*70}")
    print(f"Finished: {len(SCRIPTS) - len(failed)}/{len(SCRIPTS)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
