"""
utils.py
--------
Shared data structures and utility functions used by both RRT and RRT*.

Contents:
  Node                  — tree node with (x, y, parent, cost)
  dist()                — Euclidean distance between two nodes or (x,y) tuples
  NearestNeighborIndex  — KDTree-backed spatial index for nearest/radius queries
  rewire_radius()       — computes the shrinking neighbourhood radius for RRT*
  steer()               — steps from one point toward another by step_size
  extract_path()        — walks parent pointers to recover the path from root
"""

import math
import numpy as np
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class Node:
    """
    A single node in the RRT / RRT* tree.

    Attributes
    ----------
    x, y   : float — position in 2D space
    parent : Node or None — parent node (None only for the root)
    cost   : float — cumulative Euclidean distance from the root to this node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None   # Node or None
        self.cost = 0.0      # cumulative cost from root

    def __repr__(self):
        parent_xy = (
            f"({self.parent.x:.2f},{self.parent.y:.2f})"
            if self.parent is not None else "None"
        )
        return (
            f"Node(x={self.x:.2f}, y={self.y:.2f}, "
            f"cost={self.cost:.3f}, parent={parent_xy})"
        )


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def dist(a, b):
    """
    Euclidean distance between two points.

    Parameters
    ----------
    a, b : Node or (x, y) tuple

    Returns
    -------
    float
    """
    ax, ay = (a.x, a.y) if hasattr(a, 'x') else a
    bx, by = (b.x, b.y) if hasattr(b, 'x') else b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


# ---------------------------------------------------------------------------
# Nearest-neighbor spatial index
# ---------------------------------------------------------------------------

class NearestNeighborIndex:
    """
    KDTree-backed spatial index for efficient nearest-neighbor and
    radius queries over a set of Nodes.

    insert() rebuilds the KDTree on every call — acceptable for trees up to
    ~5,000 nodes (the experiment range). For larger trees, batch-rebuild every
    K inserts.
    """

    def __init__(self):
        self._nodes = []    # list of Node objects (same order as _coords)
        self._coords = []   # list of [x, y] — fed to KDTree
        self._kd = None     # scipy KDTree or None (empty)

    def insert(self, node):
        """
        Add a node to the index and rebuild the KDTree.

        Parameters
        ----------
        node : Node
        """
        self._nodes.append(node)
        self._coords.append([node.x, node.y])
        self._kd = KDTree(self._coords)

    def nearest(self, x, y):
        """
        Return the node closest to (x, y).

        Parameters
        ----------
        x, y : float

        Returns
        -------
        Node
        """
        if self._kd is None:
            raise RuntimeError("NearestNeighborIndex is empty — call insert() first.")
        _, idx = self._kd.query([x, y])
        return self._nodes[idx]

    def within_radius(self, x, y, r):
        """
        Return all nodes within Euclidean radius r of (x, y).

        Parameters
        ----------
        x, y : float
        r    : float — search radius

        Returns
        -------
        list of Node
        """
        if self._kd is None:
            return []
        idxs = self._kd.query_ball_point([x, y], r)
        return [self._nodes[i] for i in idxs]


# ---------------------------------------------------------------------------
# Rewiring radius (Theorem 38, Karaman & Frazzoli 2011)
# ---------------------------------------------------------------------------

def rewire_radius(n_nodes, free_vol, d=2, step_size=0.5):
    """
    Compute the neighbourhood radius used by RRT* ChooseParent and Rewire.

    Formula (Theorem 38):
        gamma = 1.1 * (2 * (1 + 1/d) * free_vol / vol_unit_ball) ^ (1/d)
        radius = gamma * (ln(n) / n) ^ (1/d)

    The radius shrinks as the tree grows, keeping the expected number of
    neighbours O(log n). A 10% safety margin on gamma ensures the theoretical
    condition is met.

    The returned radius is clamped to be at least step_size so it never
    collapses to zero for small n.

    Parameters
    ----------
    n_nodes   : int   — current number of nodes in the tree
    free_vol  : float — collision-free area estimate (from env.free_volume())
    d         : int   — space dimension (2 for 2D planning)
    step_size : float — minimum radius floor

    Returns
    -------
    float
    """
    if n_nodes < 2:
        return step_size

    unit_ball_vol = math.pi  # area of unit disc in 2D
    gamma = (2.0 * (1.0 + 1.0 / d) * (free_vol / unit_ball_vol)) ** (1.0 / d)
    gamma *= 1.1  # 10% safety margin (paper: must exceed theoretical minimum)

    radius = gamma * (math.log(n_nodes) / n_nodes) ** (1.0 / d)
    return max(radius, step_size)  # never let radius drop below step_size


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

def steer(from_node, to_point, step_size):
    """
    Move from from_node toward to_point by at most step_size.

    If the distance is smaller than step_size, returns a node at to_point
    exactly.

    Parameters
    ----------
    from_node  : Node — origin
    to_point   : (x, y) tuple — target direction
    step_size  : float

    Returns
    -------
    Node
        New node (parent and cost are NOT set here — caller sets them).
    """
    tx, ty = to_point
    d = dist(from_node, (tx, ty))
    if d < 1e-10:
        return Node(from_node.x, from_node.y)
    ratio = min(step_size / d, 1.0)
    nx = from_node.x + ratio * (tx - from_node.x)
    ny = from_node.y + ratio * (ty - from_node.y)
    return Node(nx, ny)


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------

def extract_path(goal_node):
    """
    Walk parent pointers from goal_node back to the root to recover the path.

    Parameters
    ----------
    goal_node : Node

    Returns
    -------
    list of (float, float)
        Sequence of (x, y) positions from start to goal.
    """
    path = []
    node = goal_node
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    return list(reversed(path))


# ---------------------------------------------------------------------------
# Children helper (used by propagate_cost in rrt_star.py)
# ---------------------------------------------------------------------------

def get_children(node, tree):
    """
    Return all direct children of node in the tree.

    Uses identity comparison (``is``) rather than equality so that two
    different nodes at the same coordinates are treated as distinct.

    Parameters
    ----------
    node : Node
    tree : list of Node

    Returns
    -------
    list of Node
    """
    return [n for n in tree if n.parent is node]
