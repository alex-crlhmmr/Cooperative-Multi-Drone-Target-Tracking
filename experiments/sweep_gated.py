#!/usr/bin/env python3
"""Run multiple short gated experiments sequentially and compare.

Each experiment runs for --steps (default 1M) on cluster spawn.
Results are compared at the end.

Usage:
    python -m experiments.sweep_gated
    python -m experiments.sweep_gated --steps 500000 --num-envs 8
"""
import argparse
import subprocess
import os
import numpy as np
import sys


# Define experiments: (name, extra_args)
EXPERIMENTS = {
    # A: Baseline — inverted gating, default rewards
    "A_baseline_gated": [
        "--max-rl-authority", "1.0", "--min-rl-authority", "0.3",
        "--w-sep", "0.3", "--w-dist", "0.5", "--w-cov", "1.0",
    ],
    # B: High spread reward, low distance
    "B_high_spread": [
        "--max-rl-authority", "1.0", "--min-rl-authority", "0.3",
        "--w-sep", "2.0", "--w-dist", "0.1", "--w-cov", "1.0",
    ],
    # C: Pure RL (no gating — min=max=1.0)
    "C_pure_rl": [
        "--max-rl-authority", "1.0", "--min-rl-authority", "1.0",
        "--w-sep", "1.0", "--w-dist", "0.2", "--w-cov", "1.0",
    ],
    # D: High RL authority + spread reward
    "D_high_authority_spread": [
        "--max-rl-authority", "1.0", "--min-rl-authority", "0.5",
        "--w-sep", "1.0", "--w-dist", "0.2", "--w-cov", "1.0",
    ],
    # E: Pure RL + very high spread + low cov (spread first, track later)
    "E_spread_first": [
        "--max-rl-authority", "1.0", "--min-rl-authority", "1.0",
        "--w-sep", "3.0", "--w-dist", "0.0", "--w-cov", "0.5",
    ],
}


def run_experiment(name, base_args, extra_args):
    save_path = os.path.join(base_args["output_dir"], name)
    cmd = [
        sys.executable, "-m", "experiments.train_gated",
        "--steps", str(base_args["steps"]),
        "--save-path", save_path,
        "--num-envs", str(base_args["num_envs"]),
        "--gamma", "0.995",
        "--traj", "evasive",
        "--spawn-mode", "cluster",
        "--gate-threshold", "500",
        "--tag", name,
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    print(f"  Args: {' '.join(extra_args)}")
    print(f"  Save: {save_path}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def load_eval_csv(path):
    """Load eval_results.csv and return last 5 eval tr(P) values."""
    csv_path = os.path.join(path, "eval_results.csv")
    if not os.path.exists(csv_path):
        return None

    steps, tr_Ps, rmses = [], [], []
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                steps.append(int(parts[0]))
                tr_Ps.append(float(parts[1 + 1]))  # tr_P is 3rd column
                rmses.append(float(parts[1 + 2]))   # rmse is 4th column

    return {
        "steps": steps,
        "tr_Ps": tr_Ps,
        "rmses": rmses,
        "last5_tr_P": np.mean(tr_Ps[-5:]) if len(tr_Ps) >= 5 else np.mean(tr_Ps) if tr_Ps else float("nan"),
        "last5_rmse": np.mean(rmses[-5:]) if len(rmses) >= 5 else np.mean(rmses) if rmses else float("nan"),
        "best_tr_P": min(tr_Ps) if tr_Ps else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Steps per experiment (default 1M for quick signal)")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="output/sweep_gated")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of experiment names to run (e.g. A_baseline_gated,C_pure_rl)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_args = {
        "steps": args.steps,
        "num_envs": args.num_envs,
        "output_dir": args.output_dir,
    }

    # Filter experiments if --only specified
    experiments = EXPERIMENTS
    if args.only:
        selected = set(args.only.split(","))
        experiments = {k: v for k, v in EXPERIMENTS.items() if k in selected}

    print(f"\n{'#'*60}")
    print(f"  GATED EXPERIMENT SWEEP")
    print(f"  {len(experiments)} experiments x {args.steps/1e6:.1f}M steps each")
    print(f"  Cluster spawn (±2m), evasive trajectory")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*60}\n")

    # Run sequentially
    for name, extra_args in experiments.items():
        run_experiment(name, base_args, extra_args)

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SWEEP RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<30} {'Last5 tr(P)':>12} {'Best tr(P)':>12} {'Last5 RMSE':>12}")
    print(f"  {'-'*70}")

    for name in experiments:
        save_path = os.path.join(args.output_dir, name)
        result = load_eval_csv(save_path)
        if result:
            print(f"  {name:<30} {result['last5_tr_P']:>12.1f} {result['best_tr_P']:>12.1f} {result['last5_rmse']:>12.1f}")
        else:
            print(f"  {name:<30} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    print(f"\n  Baseline cluster reference: tr(P) = 2,538,214")
    print(f"  Baseline normal reference:  tr(P) = 9.6")
    print()


if __name__ == "__main__":
    main()
