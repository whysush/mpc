"""Global A* path planner (grid-based) layered on top of the local MPC.

The MPC alone regulates directly to a goal pose, so a purely local optimizer can
stall in a local minimum whenever an obstacle sits across the straight line to
the goal (the classic long-wall failure). This module adds a *global* layer:

    A*  -> a coarse, collision-free route on an inflated occupancy grid
    MPC -> tracks a moving "carrot" along that route, and does the fine
           kinodynamic + dynamic-obstacle avoidance

Static vs dynamic obstacles are handled by the same code:

    * static scene  -> the grid never changes, so we plan once per segment.
    * dynamic scene -> the occupancy grid is rebuilt from the obstacles'
      *current* pose and A* is re-run periodically (see GlobalPlanner.plan with
      replanning driven by run_mpc). The grid inflates dynamic obstacles a bit
      extra to cover the motion between replans; the MPC still owns the exact,
      time-varying avoidance via its predicted-pose constraints.
"""

import heapq
import math

import numpy as np

from config import ROBOT_RADIUS_M, XLIM, YLIM
from controller import SAFE_MARGIN

# --- grid / planner tuning ------------------------------------------------
GRID_RES = 0.25            # occupancy cell size [m]
DYN_INFLATE = 0.05         # extra inflation for moving obstacles [m]
                           # (small: obstacles are slow and we replan often; a
                           # larger value seals corridors in dense scenes)
LOOKAHEAD = 2.5            # carrot distance along the path ahead of robot [m]
_SQRT2 = math.sqrt(2.0)

# 8-connected neighborhood: (di, dj, step_cost, is_diagonal)
_NEIGHBORS = [
    (1, 0, 1.0, False), (-1, 0, 1.0, False),
    (0, 1, 1.0, False), (0, -1, 1.0, False),
    (1, 1, _SQRT2, True), (1, -1, _SQRT2, True),
    (-1, 1, _SQRT2, True), (-1, -1, _SQRT2, True),
]


class OccupancyGrid:
    """Inflated boolean occupancy grid over the arena (XLIM x YLIM).

    A cell is blocked if its center is within `inflate` of any obstacle at
    time `t`, so a path through free cell centers keeps the robot clear.
    """

    def __init__(self, obstacles, t=0.0, res=GRID_RES):
        self.res = res
        self.x0, self.y0 = XLIM[0], YLIM[0]
        self.nx = max(1, int(math.ceil((XLIM[1] - XLIM[0]) / res)))
        self.ny = max(1, int(math.ceil((YLIM[1] - YLIM[0]) / res)))
        # half-diagonal of a cell so a free cell center truly clears the box
        base = ROBOT_RADIUS_M + SAFE_MARGIN + 0.5 * res * _SQRT2
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        for o in obstacles:
            inflate = base + (DYN_INFLATE if o.is_dynamic else 0.0)
            self._stamp(o, t, inflate)

    def _stamp(self, o, t, inflate):
        # only test cells inside the obstacle's bounding circle + inflation
        cx, cy, _ = o.pose_at(t)
        reach = 0.5 * math.hypot(o.w, o.h) + inflate
        imin, jmin = self.world_to_cell(cx - reach, cy - reach)
        imax, jmax = self.world_to_cell(cx + reach, cy + reach)
        for i in range(max(0, imin), min(self.nx, imax + 1)):
            for j in range(max(0, jmin), min(self.ny, jmax + 1)):
                if self.blocked[i, j]:
                    continue
                px, py = self.cell_to_world(i, j)
                if o.signed_distance(px, py, math, t) < inflate:
                    self.blocked[i, j] = True

    def world_to_cell(self, x, y):
        return (int((x - self.x0) / self.res), int((y - self.y0) / self.res))

    def cell_to_world(self, i, j):
        return (self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res)

    def in_bounds(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def free(self, i, j):
        return self.in_bounds(i, j) and not self.blocked[i, j]

    def nearest_free(self, i, j, max_ring=8):
        """Snap a (possibly blocked) cell to the closest free cell via ring
        search — the robot/goal can sit inside the inflation band."""
        if self.free(i, j):
            return (i, j)
        for r in range(1, max_ring + 1):
            best = None
            best_d = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue  # only the ring at radius r
                    ni, nj = i + di, j + dj
                    if self.free(ni, nj):
                        d = di * di + dj * dj
                        if best_d is None or d < best_d:
                            best_d, best = d, (ni, nj)
            if best is not None:
                return best
        return None

    def line_free(self, ax, ay, bx, by):
        """True if the straight segment (world coords) stays on free cells."""
        dist = math.hypot(bx - ax, by - ay)
        steps = max(1, int(dist / (0.5 * self.res)))
        for s in range(steps + 1):
            u = s / steps
            i, j = self.world_to_cell(ax + u * (bx - ax), ay + u * (by - ay))
            if not self.free(i, j):
                return False
        return True


def astar(grid, start_xy, goal_xy):
    """A* on the inflated grid. Returns a list of world-space (x, y) waypoints
    from start to goal (inclusive), or None if no route exists."""
    si, sj = grid.nearest_free(*grid.world_to_cell(*start_xy))
    gi, gj = grid.nearest_free(*grid.world_to_cell(*goal_xy)) or (None, None)
    if gi is None or (si, sj) == (None, None):
        return None

    def heuristic(i, j):
        return math.hypot(i - gi, j - gj)

    open_heap = [(heuristic(si, sj), 0.0, (si, sj))]
    came_from = {}
    g_score = {(si, sj): 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == (gi, gj):
            return grid_path_to_world(grid, came_from, cur, start_xy, goal_xy)
        closed.add(cur)
        ci, cj = cur
        for di, dj, step, diag in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not grid.free(ni, nj):
                continue
            if diag and not (grid.free(ci + di, cj) and grid.free(ci, cj + dj)):
                continue  # forbid corner cutting
            ng = g + step
            if ng < g_score.get((ni, nj), float("inf")):
                g_score[(ni, nj)] = ng
                came_from[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + heuristic(ni, nj), ng, (ni, nj)))
    return None


def grid_path_to_world(grid, came_from, end, start_xy, goal_xy):
    cells = [end]
    while cells[-1] in came_from:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    pts = [start_xy] + [grid.cell_to_world(i, j) for i, j in cells] + [goal_xy]
    return smooth_path(grid, pts)


def smooth_path(grid, pts):
    """Line-of-sight 'string pulling': drop intermediate points whenever the
    segment skipping them stays collision-free. Turns the grid staircase into
    a few long straight legs the MPC can track smoothly."""
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    anchor = 0
    for i in range(2, len(pts)):
        ax, ay = pts[anchor]
        bx, by = pts[i]
        if not grid.line_free(ax, ay, bx, by):
            out.append(pts[i - 1])
            anchor = i - 1
    out.append(pts[-1])
    return out


class GlobalPlanner:
    """Holds the obstacle field and produces/refreshes a global A* route, then
    serves a carrot point along it for the MPC to chase."""

    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.path = None          # list of (x, y) world points
        self.grid = None

    def plan(self, start_xy, goal_xy, t=0.0):
        """(Re)plan from start_xy to goal_xy against obstacles at time t.
        Returns True on success; on failure keeps the last path (if any)."""
        self.grid = OccupancyGrid(self.obstacles, t)
        path = astar(self.grid, start_xy, goal_xy)
        if path is not None:
            self.path = path
            return True
        return False

    def carrot(self, robot_xy, lookahead=LOOKAHEAD):
        """Return (x, y, heading) a lookahead distance along the path measured
        from the robot's closest point *on the path* (projected onto the
        nearest segment, not just the nearest vertex). Heading is the local
        path tangent. Returns None if there is no path."""
        if not self.path or len(self.path) < 2:
            return None
        path = self.path
        rx, ry = robot_xy

        # 1) closest point on the polyline (project onto each segment)
        best_d = float("inf")
        seg_i = 0
        foot = path[0]
        for k in range(len(path) - 1):
            x0, y0 = path[k]
            x1, y1 = path[k + 1]
            dx, dy = x1 - x0, y1 - y0
            seg2 = dx * dx + dy * dy
            u = 0.0 if seg2 < 1e-12 else ((rx - x0) * dx + (ry - y0) * dy) / seg2
            u = max(0.0, min(1.0, u))
            px, py = x0 + u * dx, y0 + u * dy
            d = (px - rx) ** 2 + (py - ry) ** 2
            if d < best_d:
                best_d, seg_i, foot = d, k, (px, py)

        # 2) walk forward `lookahead` from the foot point along the path
        acc = 0.0
        cx, cy = foot
        for k in range(seg_i, len(path) - 1):
            x0, y0 = (cx, cy) if k == seg_i else path[k]
            x1, y1 = path[k + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if acc + seg >= lookahead:
                u = (lookahead - acc) / seg if seg > 1e-9 else 0.0
                heading = math.atan2(y1 - y0, x1 - x0)
                return (x0 + u * (x1 - x0), y0 + u * (y1 - y0), heading)
            acc += seg

        # 3) lookahead runs past the end -> aim at the final point
        ex, ey = path[-1]
        px, py = path[-2]
        return (ex, ey, math.atan2(ey - py, ex - px))
