"""
main.py
-------
Entry point for the RRT* capstone project.

Modes
-----
  demo       — run RRT and RRT* once on a chosen environment, show side-by-side plot
  experiment — run full experiment suite, save CSVs and figures

Usage
-----
    python main.py                        # demo on env_a (default)
    python main.py --mode demo --env a
    python main.py --mode demo --env b
    python main.py --mode experiment
"""

import argparse
import os
import subprocess
import sys

# Ensure the rrt_star package directory is on the path when called from outside
sys.path.insert(0, os.path.dirname(__file__))

from environment import make_env_a, make_env_b, make_env_c
from rrt         import run_rrt
from rrt_star    import run_rrt_star
from visualise   import plot_side_by_side


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

ENV_FACTORIES = {
    'a': make_env_a,
    'b': make_env_b,
    'c': make_env_c,
}


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def run_demo(env_key, n_iter=2000, seed=42):
    """
    Run RRT and RRT* once on the chosen environment and show the result.

    Parameters
    ----------
    env_key : str — 'a', 'b', or 'c'
    n_iter  : int
    seed    : int
    """
    env = ENV_FACTORIES[env_key]()
    print(f"\n--- Demo on env_{env_key} | n_iter={n_iter} | seed={seed} ---")

    print("  Running RRT  ...", end=' ', flush=True)
    rrt_result = run_rrt(env, n_iter=n_iter, step_size=0.5, goal_bias=0.05, seed=seed)
    goal_rrt, tree_rrt = rrt_result
    if goal_rrt:
        print(f"found goal, cost={goal_rrt.cost:.3f}, nodes={len(tree_rrt)}")
    else:
        print("goal NOT found")

    print("  Running RRT* ...", end=' ', flush=True)
    rrs_result = run_rrt_star(env, n_iter=n_iter, step_size=0.5, goal_bias=0.05, seed=seed)
    goal_rrs, tree_rrs = rrs_result
    if goal_rrs:
        print(f"found goal, cost={goal_rrs.cost:.3f}, nodes={len(tree_rrs)}")
    else:
        print("goal NOT found")

    print("\nOpening plot ...")
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, f"demo_env_{env_key}.png")
    plot_side_by_side(env, rrt_result, rrs_result, save_path=save_path)
    print(f"  Figure saved to {save_path}")


# ---------------------------------------------------------------------------
# Experiment mode
# ---------------------------------------------------------------------------

def run_experiment_mode():
    """Run the full experiment pipeline (experiments.py __main__)."""
    exp_script = os.path.join(os.path.dirname(__file__), 'experiments.py')
    print(f"\nRunning experiment pipeline: {exp_script}")
    result = subprocess.run(
        [sys.executable, exp_script],
        cwd=os.path.dirname(__file__)
    )
    if result.returncode != 0:
        print("Experiment pipeline exited with errors.")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='RRT* Capstone — CS5100 Northeastern University'
    )
    parser.add_argument(
        '--mode', choices=['demo', 'experiment'],
        default='demo',
        help='demo: one-shot visualisation | experiment: full study'
    )
    parser.add_argument(
        '--env', choices=['a', 'b', 'c'],
        default='a',
        help='Environment preset (only used in demo mode)'
    )
    parser.add_argument(
        '--n_iter', type=int, default=2000,
        help='Iterations for demo mode (default: 2000)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for demo mode (default: 42)'
    )

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo(args.env, n_iter=args.n_iter, seed=args.seed)
    elif args.mode == 'experiment':
        run_experiment_mode()


if __name__ == '__main__':
    main()
