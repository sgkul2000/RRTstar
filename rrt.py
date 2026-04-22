"""
rrt.py
------
Baseline Rapidly-exploring Random Tree (RRT) planner.

RRT builds a tree from the start by randomly sampling the free space,
steering toward each sample, and adding new nodes — stopping as soon as
the goal is reached for the first time.  It does NOT improve the path
after the first solution (that is why it is suboptimal).

Reference: LaValle (1998). Rapidly-exploring random trees: a new tool for
path planning. TR 98-11, Iowa State University.
"""

import numpy as np

from utils import Node, NearestNeighborIndex, dist, steer, extract_path


def run_rrt(env, n_iter=2000, step_size=0.5, goal_bias=0.05, seed=None):
    """
    Run the standard RRT algorithm.

    Algorithm outline
    -----------------
    1. Initialise tree with root at env.start.
    2. For each iteration:
       a. Sample a random free point (with probability goal_bias, use the
          goal directly — "goal biasing").
       b. Find the nearest node in the tree.
       c. Steer from nearest toward the sample by step_size.
       d. If the new segment is collision-free, add the new node.
       e. If the new node is inside the goal region, return immediately.
    3. Return (None, tree) if the goal was not reached within n_iter.

    Parameters
    ----------
    env       : Environment
    n_iter    : int   — maximum number of iterations
    step_size : float — maximum steering distance per step
    goal_bias : float — probability of sampling the goal directly (0–1)
    seed      : int or None — random seed for reproducibility

    Returns
    -------
    (goal_node_or_None, tree)
        goal_node_or_None : Node — the first node that reached the goal,
                            or None if the goal was not reached.
        tree              : list of Node — all nodes added to the tree.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialise tree with root at start
    root = Node(*env.start)
    root.cost = 0.0
    tree = [root]

    index = NearestNeighborIndex()
    index.insert(root)

    for _ in range(n_iter):
        # --- Sample ---
        if np.random.random() < goal_bias:
            x_rand = env.goal
        else:
            x_rand = env.sample_free()

        # --- Nearest ---
        x_nearest = index.nearest(*x_rand)

        # --- Steer ---
        x_new = steer(x_nearest, x_rand, step_size)

        # --- Collision check ---
        if not env.is_collision_free(
            (x_nearest.x, x_nearest.y), (x_new.x, x_new.y)
        ):
            continue

        # --- Add node ---
        x_new.parent = x_nearest
        x_new.cost = x_nearest.cost + dist(x_nearest, x_new)
        tree.append(x_new)
        index.insert(x_new)

        # --- Goal check — RRT stops at first solution ---
        if env.in_goal(x_new):
            return x_new, tree

    return None, tree
