import math
import pygame
from geometry import world_to_screen, scale
from config import (ROBOT_RADIUS_M, C_ROBOT, C_ROBOT_HIT, C_HEADING,
                    V_MAX, V_MIN, OMEGA_MAX, OMEGA_MIN)


def unicycle_step(state, control, dt, sin=math.sin, cos=math.cos):
    x, y, theta = state[0], state[1], state[2]
    v, omega = control[0], control[1]
    x_new = x + v * cos(theta) * dt
    y_new = y + v * sin(theta) * dt
    theta_new = theta + omega * dt
    return [x_new, y_new, theta_new]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.r = ROBOT_RADIUS_M
        self.collided = False

    @property
    def state(self):
        return [self.x, self.y, self.theta]

    def apply(self, v, omega, dt):
        v = clamp(v, V_MIN, V_MAX)
        omega = clamp(omega, OMEGA_MIN, OMEGA_MAX)
        self.x, self.y, self.theta = unicycle_step(self.state, [v, omega], dt)

    def check_collision(self, obstacles, t=0.0):
        self.collided = any(
            o.collides_circle(self.x, self.y, self.r, t) for o in obstacles
        )
        return self.collided

    def draw(self, surface):
        color = C_ROBOT_HIT if self.collided else C_ROBOT
        cx, cy = world_to_screen(self.x, self.y)
        pygame.draw.circle(surface, color, (cx, cy), scale(self.r))
        hx = self.x + self.r * math.cos(self.theta)
        hy = self.y + self.r * math.sin(self.theta)
        pygame.draw.line(surface, C_HEADING, (cx, cy),
                         world_to_screen(hx, hy), 3)