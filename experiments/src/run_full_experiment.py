"""
Run the complete adversarial attack experiment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiment_framework import BracketCountingExperiment
from datetime import datetime

print("="*60)
print("FULL EXPERIMENT RUN")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize
exp = BracketCountingExperiment()
exp.load_data()

# Run all phases
print("\n" + "="*60)
print("PHASE 1: CLEAN BASELINE")
print("="*60)
baseline_stats = exp.run_clean_baseline()

print("\n" + "="*60)
print("PHASE 2: RANDOM ATTACKS")
print("="*60)
random_stats = exp.run_random_attacks([1, 2, 3, 5])

print("\n" + "="*60)
print("PHASE 3: CIRCUIT-TARGETED ATTACKS")
print("="*60)
circuit_stats = exp.run_circuit_attacks([1, 2, 3, 5])

# Save all results
exp.save_results("full_experiment_results.csv")

# Analyze
analysis = exp.analyze()

# Final summary
print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total results recorded: {len(exp.results)}")
print(f"Results saved to: data/full_experiment_results.csv")
