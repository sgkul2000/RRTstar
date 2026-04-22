"""
rrt_star.py
-----------
RRT* — the asymptotically optimal extension of RRT.

RRT* adds two procedures to the standard RRT loop:
  1. ChooseParent — instead of connecting x_new to the nearest node, find
     the collision-free node within the rewiring radius that gives the
     lowest cumulative cost from the root.
  2. Rewire — after adding x_new, check whether any nearby node can reach
     its own subtree at a lower cost by routing through x_new.  If so,
     re-parent that node and propagate the cost improvement downward.

Unlike RRT, RRT* does NOT stop at the first solution.  It continues for
all n_iter iterations, keeping track of the best goal node found so far.
This allows the path cost to monotonically decrease with more iterations.

Reference: Karaman & Frazzoli (2011). Sampling-based algorithms for
optimal motion planning. IJRR 30(7): 846–894.
"""

import numpy as np
from collections import deque

from utils import Node, NearestNeighborIndex, dist, steer, get_children, rewire_radius


# ---------------------------------------------------------------------------
# ChooseParent
# ---------------------------------------------------------------------------

def choose_parent(x_new, near_nodes, env):
    """
    Set x_new's parent to the nearby node that yields the lowest total
    cost from the root, subject to collision-free connectivity.

    If no collision-free nearby node exists, x_new is returned unchanged
    (parent remains None, cost remains inf).

    Parameters
    ----------
    x_new      : Node — the newly steered node (modified in place)
    near_nodes : list of Node — candidate parents within rewiring radius
    env        : Environment — used for collision checking

    Returns
    -------
    Node
        x_new with .parent and .cost updated.
    """
    best_parent = None
    best_cost = float('inf')

    for node in near_nodes:
        d = dist(node, x_new)
        candidate_cost = node.cost + d
        if candidate_cost < best_cost and env.is_collision_free(
            (node.x, node.y), (x_new.x, x_new.y)
        ):
            best_cost = candidate_cost
            best_parent = node

    x_new.parent = best_parent
    x_new.cost = best_cost
    return x_new


# ---------------------------------------------------------------------------
# Cost propagation
# ---------------------------------------------------------------------------

def propagate_cost(node, tree):
    """
    Propagate a cost update from node to ALL of its descendants using BFS.

    This must be called after rewiring a node so that every descendant's
    cumulative cost is consistent with the new parent chain.

    Parameters
    ----------
    node : Node — the node whose cost has just been updated (the BFS root)
    tree : list of Node — the full tree (used to find children)
    """
    queue = deque([node])
    while queue:
        current = queue.popleft()
        for child in get_children(current, tree):
            child.cost = current.cost + dist(current, child)
            queue.append(child)


# ---------------------------------------------------------------------------
# Rewire
# ---------------------------------------------------------------------------

def rewire(x_new, near_nodes, env, tree):
    """
    For each nearby node, check whether routing through x_new reduces its
    cost.  If yes, update the node's parent, recompute its cost, and
    propagate the change to all descendants.

    Parameters
    ----------
    x_new      : Node — the newly added node
    near_nodes : list of Node — nodes within the rewiring radius
    env        : Environment — used for collision checking
    tree       : list of Node — the full tree (needed by propagate_cost)
    """
    for node in near_nodes:
        if node is x_new.parent or node is x_new:
            continue  # skip x_new's own parent and itself

        d = dist(x_new, node)
        new_cost = x_new.cost + d

        if new_cost < node.cost and env.is_collision_free(
            (x_new.x, x_new.y), (node.x, node.y)
        ):
            node.parent = x_new
            node.cost = new_cost
            propagate_cost(node, tree)  # update ALL descendants


# ---------------------------------------------------------------------------
# Main RRT* loop
# ---------------------------------------------------------------------------

def run_rrt_star(env, n_iter=2000, step_size=0.5, goal_bias=0.05, seed=None):
    """
    Run the RRT* algorithm.

    Algorithm outline
    -----------------
    1. Initialise tree with root at env.start.
    2. For each iteration:
       a. Sample a random free point (with goal_bias probability, use goal).
       b. Find the nearest node.
       c. Steer toward the sample.
       d. If the steered segment is collision-free:
          i.  Compute the rewiring radius for the current tree size.
          ii. ChooseParent from all nodes within radius.
          iii.If x_new has a valid parent, add it to the tree.
          iv. Rewire nearby nodes through x_new where beneficial.
       e. Track the best goal node found (lowest cost) — do NOT stop early.
    3. Return (best_goal_node_or_None, tree).

    Parameters
    ----------
    env       : Environment
    n_iter    : int   — total number of iterations (more → better path)
    step_size : float — maximum steering distance per step
    goal_bias : float — probability of sampling the goal directly (0–1)
    seed      : int or None — random seed for reproducibility

    Returns
    -------
    (best_goal_node_or_None, tree)
        best_goal_node_or_None : Node — goal node with lowest cost found,
                                 or None if the goal was not reached.
        tree                   : list of Node — all nodes in the tree.
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialise tree with root at start
    root = Node(*env.start)
    root.cost = 0.0
    tree = [root]

    index = NearestNeighborIndex()
    index.insert(root)

    best_goal_node = None
    free_vol = env.free_volume()

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

        # --- Collision check (nearest → new) ---
        if not env.is_collision_free(
            (x_nearest.x, x_nearest.y), (x_new.x, x_new.y)
        ):
            continue

        # --- Compute rewiring radius ---
        r = rewire_radius(len(tree), free_vol, d=2, step_size=step_size)

        # --- ChooseParent (RRT* step 1) ---
        near_nodes = index.within_radius(x_new.x, x_new.y, r)

        # If no near nodes found (early iterations), fall back to x_nearest
        if not near_nodes:
            near_nodes = [x_nearest]

        x_new = choose_parent(x_new, near_nodes, env)

        # If choose_parent found no valid parent, skip this sample
        if x_new.parent is None:
            continue

        # --- Add node to tree ---
        tree.append(x_new)
        index.insert(x_new)

        # --- Rewire (RRT* step 2) ---
        rewire(x_new, near_nodes, env, tree)

        # --- Track best goal ---
        if env.in_goal(x_new):
            if best_goal_node is None or x_new.cost < best_goal_node.cost:
                best_goal_node = x_new

    return best_goal_node, tree
