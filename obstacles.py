import math
import pygame
from geometry import world_to_screen
from config import (C_OBSTACLE, C_OBS_EDGE, C_DYN_OBS, C_DYN_OBS_EDGE,
                    C_PARAM, ROBOT_RADIUS_M, DT)
from utils import load_obstacles


class Obstacle:
    """Rotated rectangle that may translate (along its heading) and rotate.

    Pose at time t [s]:
        cx(t) = cx0 + v * cos(angle0) * t
        cy(t) = cy0 + v * sin(angle0) * t
        angle(t) = angle0 + omega * t

    Translation is along the *initial* heading (constant-velocity drift); the
    body also spins at rate omega. Static obstacles have v = omega = 0, so
    every time-dependent term drops out and the pose is constant.
    """

    def __init__(self, cx, cy, w, h, angle=0.0, v=0.0, omega=0.0):
        self.cx0 = cx
        self.cy0 = cy
        self.w = w
        self.h = h
        self.angle0 = angle
        self.v = v
        self.omega = omega
        # constant drift direction (fixed at the initial heading)
        self._dx = v * math.cos(angle)
        self._dy = v * math.sin(angle)

    @property
    def is_dynamic(self):
        return self.v != 0.0 or self.omega != 0.0

    def pose_at(self, t):
        return (self.cx0 + self._dx * t,
                self.cy0 + self._dy * t,
                self.angle0 + self.omega * t)

    @property
    def cx(self):
        return self.cx0

    @property
    def cy(self):
        return self.cy0

    @property
    def angle(self):
        return self.angle0

    def corners(self, t=0.0):
        cx, cy, ang = self.pose_at(t)
        hw, hh = self.w / 2.0, self.h / 2.0
        ca, sa = math.cos(ang), math.sin(ang)
        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts = []
        for lx, ly in local:
            wx = cx + lx * ca - ly * sa
            wy = cy + lx * sa + ly * ca
            pts.append((wx, wy))
        return pts

    def signed_distance(self, px, py, m, t=0.0):
        """Signed distance from point (px, py) to the box at time t.

        m is the math backend: Python's `math` (scalars, for sim/collision)
        or `casadi` (symbolics, inside the MPC). Negative inside, positive
        outside.

        t may be a plain float (sim/collision) or a CasADi expression (when
        the MPC anchors the horizon to a solve-time clock); the pose terms are
        computed with the matching backend so both cases stay differentiable.
        """
        cx = self.cx0 + self._dx * t
        cy = self.cy0 + self._dy * t
        ang = self.angle0 + self.omega * t
        dx, dy = px - cx, py - cy
        c, s = m.cos(-ang), m.sin(-ang)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        ax = m.fabs(lx) - self.w / 2.0
        ay = m.fabs(ly) - self.h / 2.0
        if m is math:
            outside = math.hypot(max(ax, 0.0), max(ay, 0.0))
            inside = min(max(ax, ay), 0.0)
        else:
            outside = m.sqrt(m.fmax(ax, 0) ** 2 + m.fmax(ay, 0) ** 2 + 1e-6)
            inside = m.fmin(m.fmax(ax, ay), 0.0)
        return outside + inside

    def collides_circle(self, px, py, r=ROBOT_RADIUS_M, t=0.0):
        return self.signed_distance(px, py, math, t) < r

    def draw(self, surface, t=0.0):
        edge = C_DYN_OBS_EDGE if self.is_dynamic else C_OBS_EDGE
        fill = C_DYN_OBS if self.is_dynamic else C_OBSTACLE
        pts = [world_to_screen(x, y) for (x, y) in self.corners(t)]
        pygame.draw.polygon(surface, fill, pts)
        pygame.draw.polygon(surface, edge, pts, 2)

    def draw_parameterization(self, surface, t=0.0):
        """Visualize the obstacle's parameterization: center, heading axis,
        and (for dynamic obstacles) the velocity vector."""
        cx, cy, ang = self.pose_at(t)
        c = world_to_screen(cx, cy)
        pygame.draw.circle(surface, C_PARAM, c, 3)
        # heading axis (half-width along local x)
        hx = cx + (self.w / 2.0) * math.cos(ang)
        hy = cy + (self.w / 2.0) * math.sin(ang)
        pygame.draw.line(surface, C_PARAM, c, world_to_screen(hx, hy), 1)
        if self.is_dynamic:
            # velocity arrow (scaled up so it's visible)
            vx = cx + self._dx * 8.0
            vy = cy + self._dy * 8.0
            pygame.draw.line(surface, C_PARAM, c, world_to_screen(vx, vy), 2)


def load_scene(scene, data_dir="."):
    """Load obstacles from a .obs file (decrypted via utils.load_obstacles)."""
    raw = load_obstacles(scene, data_dir=data_dir)
    return [
        Obstacle(o.cx, o.cy, o.width, o.height, o.angle, o.v, o.omega)
        for o in raw
    ]


def default_obstacle_field():
    """Backwards-compatible default: the static scene."""
    return load_scene("static")
