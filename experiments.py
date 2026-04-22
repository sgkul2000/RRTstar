"""
experiments.py
--------------
Runs systematic experiments comparing RRT and RRT* across three environments,
multiple iteration counts, and multiple independent trials.

Usage
-----
    python experiments.py
"""

import os
import time
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
FULL_N_VALS   = [500, 1000, 2000, 3000, 5000]

FULL_TRIALS   = 50

STEP_SIZE     = 0.5
GOAL_BIAS     = 0.05

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(algo_fn, env, n_iter, step_size, seed):
    """
    Run one trial of an algorithm and record metrics.

    Parameters
    ----------
    algo_fn    : callable — either run_rrt or run_rrt_star
    env        : Environment
    n_iter     : int — number of iterations for this trial
    step_size  : float
    seed       : int — random seed (use trial index for reproducibility)

    Returns
    -------
    dict with keys:
        cost          : float — path cost (inf if goal not reached)
        time_seconds  : float — wall-clock time in seconds
        success       : bool  — whether the goal was reached
        n_nodes       : int   — number of nodes in the final tree
    """
    t0 = time.perf_counter()
    goal_node, tree = algo_fn(
        env, n_iter=n_iter, step_size=step_size, goal_bias=GOAL_BIAS, seed=seed
    )
    elapsed = time.perf_counter() - t0

    success = goal_node is not None
    cost = goal_node.cost if success else float('inf')

    return {
        'cost':         cost,
        'time_seconds': elapsed,
        'success':      success,
        'n_nodes':      len(tree),
    }


# ---------------------------------------------------------------------------
# Full experiment sweep
# ---------------------------------------------------------------------------

def run_experiment(env_name, env, n_vals, n_trials, step_size, algo_fn):
    """
    Run n_trials independent trials for each value in n_vals.

    Parameters
    ----------
    env_name  : str — label for the environment (e.g. 'env_a')
    env       : Environment
    n_vals    : list of int — iteration counts to sweep
    n_trials  : int — number of independent trials per (n, algo) pair
    step_size : float
    algo_fn   : callable — run_rrt or run_rrt_star

    Returns
    -------
    pd.DataFrame
        Columns: env, algo, n_iter, trial, cost, time_seconds, success, n_nodes
    """
    algo_name = algo_fn.__name__  # 'run_rrt' or 'run_rrt_star'
    rows = []

    for n in n_vals:
        print(f"    {env_name} | {algo_name} | n={n} | running {n_trials} trials ...")
        for trial in range(n_trials):
            result = run_trial(algo_fn, env, n_iter=n, step_size=step_size, seed=trial)
            rows.append({
                'env':          env_name,
                'algo':         algo_name,
                'n_iter':       n,
                'trial':        trial,
                'cost':         result['cost'],
                'time_seconds': result['time_seconds'],
                'success':      result['success'],
                'n_nodes':      result['n_nodes'],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results(df, filename):
    """
    Save a results DataFrame to the results/ folder as CSV.

    Parameters
    ----------
    df       : pd.DataFrame
    filename : str — filename (e.g. 'env_a_rrt.csv')
    """
    path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def build_cost_dict(df, env_name, n_vals):
    """
    Build the results_dict format expected by plot_cost_vs_iterations.

    Only successful trials are included in the cost statistics.  Inf costs
    (failed trials) are excluded from mean/std so the plot is not distorted.

    Parameters
    ----------
    df       : pd.DataFrame — combined results for one environment
    env_name : str — used to filter
    n_vals   : list of int

    Returns
    -------
    dict  {'rrt': {...}, 'rrt_star': {...}}
    """
    results = {}
    for algo_col, key in [('run_rrt', 'rrt'), ('run_rrt_star', 'rrt_star')]:
        sub = df[(df['env'] == env_name) & (df['algo'] == algo_col)]
        mean_costs, std_costs = [], []
        for n in n_vals:
            costs = sub[sub['n_iter'] == n]['cost']
            finite = costs[np.isfinite(costs)]
            mean_costs.append(finite.mean() if len(finite) else float('nan'))
            std_costs.append(finite.std()  if len(finite) else 0.0)
        results[key] = {
            'n_vals':     n_vals,
            'mean_costs': mean_costs,
            'std_costs':  std_costs,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from environment import make_env_a, make_env_b, make_env_c
    from rrt      import run_rrt
    from rrt_star import run_rrt_star
    from visualise import plot_cost_vs_iterations

    n_vals  = FULL_N_VALS
    n_trials = FULL_TRIALS

    envs = {
        'env_a': make_env_a(),
        'env_b': make_env_b(),
        'env_c': make_env_c(),
    }

    all_dfs = []

    for env_name, env in envs.items():
        print(f"\n=== {env_name.upper()} ===")
        for algo_fn in [run_rrt, run_rrt_star]:
            df = run_experiment(env_name, env, n_vals, n_trials, STEP_SIZE, algo_fn)
            save_results(df, f"{env_name}_{algo_fn.__name__}.csv")
            all_dfs.append(df)

    # Combine and save master CSV
    combined = pd.concat(all_dfs, ignore_index=True)
    save_results(combined, 'all_results.csv')

    # Generate cost-vs-iterations plots for each environment
    print("\nGenerating figures ...")
    for env_name in envs:
        sub = combined[combined['env'] == env_name]
        cost_dict = build_cost_dict(sub, env_name, n_vals)
        plot_cost_vs_iterations(
            cost_dict,
            title=f"Cost vs Iterations — {env_name.replace('_', ' ').title()}",
            save_path=os.path.join(FIGURES_DIR, f"cost_vs_iter_{env_name}.png"),
        )

    print("\nDone.  Results in results/  |  Figures in figures/")
