"""
environment.py
--------------
Defines the 2D planning environment: map dimensions, rectangular obstacles,
start/goal positions, and sampling/collision utilities.

Three preset environments are provided as module-level factory functions:
  make_env_a() — single large central obstacle
  make_env_b() — narrow passage between two tall obstacles
  make_env_c() — maze-like with 5-6 smaller obstacles
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Environment:
    """
    2D rectangular map with axis-aligned rectangular obstacles.

    Parameters
    ----------
    width, height : float
        Map dimensions.
    obstacles : list of (x, y, w, h)
        Each obstacle is an axis-aligned rectangle with bottom-left corner
        at (x, y) and size (w, h).
    start : (float, float)
        Start position (x, y).
    goal : (float, float)
        Goal position (x, y).
    goal_radius : float
        A node is 'in goal' if its distance to goal is <= goal_radius.
    """

    def __init__(self, width, height, obstacles, start, goal, goal_radius=0.5):
        self.width = width
        self.height = height
        self.obstacles = obstacles  # list of (x, y, w, h)
        self.start = start
        self.goal = goal
        self.goal_radius = goal_radius

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_free(self, max_attempts=10000):
        """
        Return a random (x, y) point inside the map that is not inside
        any obstacle. Uses rejection sampling.

        Returns
        -------
        (float, float)
            A collision-free point sampled uniformly from free space.

        Raises
        ------
        RuntimeError
            If max_attempts rejections occur without finding a free point.
        """
        for _ in range(max_attempts):
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            if not self._point_in_any_obstacle(x, y):
                return (x, y)
        raise RuntimeError(
            f"sample_free() failed to find a free point in {max_attempts} attempts. "
            "Check that the obstacle configuration leaves some free space."
        )

    def _point_in_obstacle(self, x, y, ox, oy, ow, oh):
        """Return True if (x,y) is strictly inside rectangle (ox,oy,ow,oh)."""
        return ox <= x <= ox + ow and oy <= y <= oy + oh

    def _point_in_any_obstacle(self, x, y):
        """Return True if (x,y) is inside any obstacle."""
        return any(
            self._point_in_obstacle(x, y, ox, oy, ow, oh)
            for (ox, oy, ow, oh) in self.obstacles
        )

    # ------------------------------------------------------------------
    # Collision checking (slab method for AABB)
    # ------------------------------------------------------------------

    def is_collision_free(self, p1, p2):
        """
        Check whether the straight-line segment from p1 to p2 is free of
        obstacles. Uses the parametric slab method for axis-aligned
        rectangles: intersect the segment against the x-slab and y-slab
        of each obstacle, then check if the overlapping parameter
        interval intersects [0, 1].

        Also checks if either endpoint lies inside an obstacle.

        Parameters
        ----------
        p1, p2 : tuple (x, y) or Node
            Endpoints of the segment (can be Nodes with .x/.y or plain tuples).

        Returns
        -------
        bool
            True if the segment does not intersect any obstacle.
        """
        # Accept both Node objects and plain (x, y) tuples
        x1, y1 = (p1.x, p1.y) if hasattr(p1, 'x') else p1
        x2, y2 = (p2.x, p2.y) if hasattr(p2, 'x') else p2

        # Check endpoints themselves first
        if self._point_in_any_obstacle(x1, y1) or self._point_in_any_obstacle(x2, y2):
            return False

        dx = x2 - x1
        dy = y2 - y1

        for (ox, oy, ow, oh) in self.obstacles:
            if self._segment_intersects_rect(x1, y1, dx, dy, ox, oy, ow, oh):
                return False

        return True

    def _segment_intersects_rect(self, x1, y1, dx, dy, ox, oy, ow, oh):
        """
        Slab-method intersection test for segment (x1,y1)+(dx,dy)*t, t in [0,1]
        against axis-aligned rectangle [ox, ox+ow] x [oy, oy+oh].

        Returns True if the segment crosses (or touches) the rectangle.
        """
        t_min = 0.0
        t_max = 1.0

        # X slab: ox <= x1 + dx*t <= ox+ow
        if abs(dx) < 1e-10:
            # Segment is vertical — outside x slab means no intersection
            if x1 < ox or x1 > ox + ow:
                return False
        else:
            t1 = (ox - x1) / dx
            t2 = (ox + ow - x1) / dx
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False

        # Y slab: oy <= y1 + dy*t <= oy+oh
        if abs(dy) < 1e-10:
            if y1 < oy or y1 > oy + oh:
                return False
        else:
            t1 = (oy - y1) / dy
            t2 = (oy + oh - y1) / dy
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False

        return True  # slabs overlap in [0,1] → intersection

    # ------------------------------------------------------------------
    # Goal check
    # ------------------------------------------------------------------

    def in_goal(self, node):
        """
        Return True if node is within goal_radius of the goal position.

        Parameters
        ----------
        node : Node or (x, y)
        """
        x, y = (node.x, node.y) if hasattr(node, 'x') else node
        gx, gy = self.goal
        return ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5 <= self.goal_radius

    # ------------------------------------------------------------------
    # Free volume estimate
    # ------------------------------------------------------------------

    def free_volume(self):
        """
        Estimate the collision-free area of the map.

        Computed as: (map area - total obstacle area) * 0.85
        The 0.85 factor is a conservative estimate to account for
        overlapping obstacles and narrow gaps.

        Returns
        -------
        float
            Estimated free-space area.
        """
        obstacle_area = sum(w * h for (_, _, w, h) in self.obstacles)
        map_area = self.width * self.height
        return max((map_area - obstacle_area) * 0.85, 1.0)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax):
        """
        Draw the environment on a Matplotlib axis.

        Obstacles → filled grey rectangles with black edge.
        Start     → green filled circle.
        Goal      → red filled circle with dashed boundary circle showing goal_radius.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        """
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        for (ox, oy, ow, oh) in self.obstacles:
            rect = patches.Rectangle(
                (ox, oy), ow, oh,
                linewidth=1, edgecolor='black',
                facecolor='grey', alpha=0.7, zorder=2
            )
            ax.add_patch(rect)

        # Start (green circle)
        ax.plot(*self.start, 'go', markersize=8, zorder=5, label='Start')

        # Goal (red circle + dashed boundary)
        ax.plot(*self.goal, 'r*', markersize=10, zorder=5, label='Goal')
        goal_circle = plt.Circle(
            self.goal, self.goal_radius,
            color='red', fill=False, linestyle='--', linewidth=1.5, zorder=3
        )
        ax.add_patch(goal_circle)


# ---------------------------------------------------------------------------
# Preset environments
# ---------------------------------------------------------------------------

def make_env_a():
    """
    Environment A — single large obstacle in the centre.

    Map: 10 × 10. One 3×4 obstacle centred at (3.5, 3).
    Start: (1, 1). Goal: (9, 9).
    This is the simplest environment; RRT can usually find a path quickly.

    Returns
    -------
    Environment
    """
    width, height = 10.0, 10.0
    obstacles = [
        (3.5, 3.0, 3.0, 4.0),  # central blocker
    ]
    start = (1.0, 1.0)
    goal = (9.0, 9.0)
    return Environment(width, height, obstacles, start, goal, goal_radius=0.5)


def make_env_b():
    """
    Environment B — narrow passage.

    Two tall obstacles create a tight gap that the planner must thread through.
    Map: 10 × 10.
    Start: (1, 5). Goal: (9, 5) — both on the horizontal midline.
    The passage is only ~1 unit wide, making this challenging for RRT.

    Returns
    -------
    Environment
    """
    width, height = 10.0, 10.0
    obstacles = [
        (4.5, 0.0, 1.0, 4.0),   # lower wall piece
        (4.5, 5.0, 1.0, 5.0),   # upper wall piece (gap is y in [4,5])
    ]
    start = (1.0, 5.0)
    goal = (9.0, 5.0)
    return Environment(width, height, obstacles, start, goal, goal_radius=0.5)


def make_env_c():
    """
    Environment C — maze-like with 5 smaller obstacles creating corridors.

    Map: 10 × 10.
    Start: (0.5, 0.5). Goal: (9.5, 9.5).
    Corridors force the planner to navigate around multiple obstacles.

    Returns
    -------
    Environment
    """
    width, height = 10.0, 10.0
    obstacles = [
        (1.5, 1.5, 2.5, 1.0),   # row 1 left
        (5.0, 1.0, 2.0, 2.5),   # row 1 right
        (2.0, 4.5, 2.5, 1.0),   # row 2
        (6.0, 5.0, 2.0, 1.5),   # row 3
        (1.5, 7.5, 3.0, 1.0),   # row 4 left
        (6.5, 7.0, 2.0, 2.0),   # row 4 right
    ]
    start = (0.5, 0.5)
    goal = (9.5, 9.5)
    return Environment(width, height, obstacles, start, goal, goal_radius=0.5)
