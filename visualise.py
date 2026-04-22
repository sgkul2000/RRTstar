"""
visualise.py
------------
Matplotlib visualisation utilities for RRT and RRT* results.

Functions
---------
plot_tree_and_path     — draw the full tree + solution path on one axis
plot_cost_vs_iterations — mean ± std cost curves for both algorithms
plot_side_by_side      — two-panel comparison of RRT vs RRT* on same env
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import extract_path


# ---------------------------------------------------------------------------
# Single-run tree + path plot
# ---------------------------------------------------------------------------

def plot_tree_and_path(env, tree, path_nodes, title, ax=None):
    """
    Visualise the search tree and the found path.

    Draws:
      • Environment (obstacles, start, goal) via env.plot()
      • All tree edges as thin light-grey lines
      • All tree nodes as small grey dots
      • The solution path as a thick red line

    Parameters
    ----------
    env        : Environment
    tree       : list of Node — every node in the tree
    path_nodes : list of (x, y) — the solution path, root to goal
                 (pass [] or None if no path was found)
    title      : str — axis title
    ax         : matplotlib.axes.Axes or None
                 If None, a new figure is created and shown.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Environment (obstacles, start, goal)
    env.plot(ax)

    # Tree edges (light grey)
    for node in tree:
        if node.parent is not None:
            ax.plot(
                [node.parent.x, node.x],
                [node.parent.y, node.y],
                color='lightgrey', linewidth=0.5, zorder=1
            )

    # Tree nodes (small grey dots)
    xs = [n.x for n in tree]
    ys = [n.y for n in tree]
    ax.scatter(xs, ys, s=2, color='grey', zorder=2)

    # Solution path (thick red)
    if path_nodes:
        px = [p[0] for p in path_nodes]
        py = [p[1] for p in path_nodes]
        ax.plot(px, py, color='red', linewidth=2.5, zorder=4, label='Path')

    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper left', fontsize=8)

    if standalone:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Cost vs iterations line plot
# ---------------------------------------------------------------------------

def plot_cost_vs_iterations(results_dict, title, save_path=None):
    """
    Plot mean path cost ± std vs number of iterations for multiple algorithms.

    Parameters
    ----------
    results_dict : dict
        Format::

            {
              'rrt':      {'n_vals': [...], 'mean_costs': [...], 'std_costs': [...]},
              'rrt_star': {'n_vals': [...], 'mean_costs': [...], 'std_costs': [...]}
            }

        Keys should match the algorithm names you want in the legend.
    title     : str — figure title
    save_path : str or None — if provided, save the figure here (PNG)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        'rrt':      {'color': 'steelblue',  'label': 'RRT',   'ls': '--'},
        'rrt_star': {'color': 'darkorange', 'label': 'RRT*',  'ls': '-'},
    }

    for algo, data in results_dict.items():
        style = styles.get(algo, {'color': 'black', 'label': algo, 'ls': '-'})
        n_vals = data['n_vals']
        means  = np.array(data['mean_costs'])
        stds   = np.array(data['std_costs'])

        ax.plot(n_vals, means, color=style['color'], linestyle=style['ls'],
                linewidth=2, marker='o', markersize=5, label=style['label'])
        ax.fill_between(
            n_vals, means - stds, means + stds,
            color=style['color'], alpha=0.15
        )

    ax.set_xlabel('Iterations (N)', fontsize=12)
    ax.set_ylabel('Path Cost (Euclidean length)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Side-by-side RRT vs RRT* comparison
# ---------------------------------------------------------------------------

def plot_side_by_side(env, rrt_result, rrtstar_result, save_path=None):
    """
    Two-panel figure comparing a single RRT run (left) and RRT* run (right)
    on the same environment.

    Parameters
    ----------
    env           : Environment
    rrt_result    : (goal_node_or_None, tree) — output of run_rrt()
    rrtstar_result: (goal_node_or_None, tree) — output of run_rrt_star()
    save_path     : str or None — if provided, save the figure here (PNG)
    """
    rrt_goal, rrt_tree = rrt_result
    rrs_goal, rrs_tree = rrtstar_result

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left — RRT
    rrt_path = extract_path(rrt_goal) if rrt_goal else []
    rrt_cost = rrt_goal.cost if rrt_goal else float('inf')
    plot_tree_and_path(
        env, rrt_tree, rrt_path,
        title=f"RRT  |  nodes={len(rrt_tree)}  |  cost={rrt_cost:.2f}",
        ax=axes[0]
    )

    # Right — RRT*
    rrs_path = extract_path(rrs_goal) if rrs_goal else []
    rrs_cost = rrs_goal.cost if rrs_goal else float('inf')
    plot_tree_and_path(
        env, rrs_tree, rrs_path,
        title=f"RRT*  |  nodes={len(rrs_tree)}  |  cost={rrs_cost:.2f}",
        ax=axes[1]
    )

    fig.suptitle("RRT vs RRT* — path cost comparison", fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
