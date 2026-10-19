import sys
import math
import pygame

from config import (WINDOW_PX, FPS, DT, ARENA_M, START_POSE, GOAL_M,
                    V_STEP, OMEGA_STEP, V_MAX, V_MIN, OMEGA_MAX, OMEGA_MIN,
                    C_BG, C_GRID, C_GOAL, C_TRAIL, C_TEXT)
from geometry import world_to_screen
from obstacles import default_obstacle_field
from robot import Robot, clamp


def draw_grid(surface):
    for m in range(int(ARENA_M) + 1):
        x0, y0 = world_to_screen(m, 0)
        x1, y1 = world_to_screen(m, ARENA_M)
        pygame.draw.line(surface, C_GRID, (x0, y0), (x1, y1), 1)
        hx0, hy0 = world_to_screen(0, m)
        hx1, hy1 = world_to_screen(ARENA_M, m)
        pygame.draw.line(surface, C_GRID, (hx0, hy0), (hx1, hy1), 1)


def draw_goal(surface):
    gx, gy = world_to_screen(*GOAL_M)
    s = 10
    pygame.draw.line(surface, C_GOAL, (gx - s, gy - s), (gx + s, gy + s), 3)
    pygame.draw.line(surface, C_GOAL, (gx - s, gy + s), (gx + s, gy - s), 3)


def draw_trail(surface, trail):
    if len(trail) > 1:
        pts = [world_to_screen(x, y) for (x, y) in trail]
        pygame.draw.lines(surface, C_TRAIL, False, pts, 2)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PX, WINDOW_PX))
    pygame.display.set_caption("MPC Project - Week 1: manual driving")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    obstacles = default_obstacle_field()
    robot = Robot(*START_POSE)
    trail = []
    v_cmd = 0.0
    omega_cmd = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    v_cmd, omega_cmd = 0.0, 0.0
                elif event.key == pygame.K_r:
                    robot = Robot(*START_POSE)
                    trail = []
                    v_cmd, omega_cmd = 0.0, 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            v_cmd = clamp(v_cmd + V_STEP, V_MIN, V_MAX)
        if keys[pygame.K_DOWN]:
            v_cmd = clamp(v_cmd - V_STEP, V_MIN, V_MAX)
        if keys[pygame.K_LEFT]:
            omega_cmd = clamp(omega_cmd + OMEGA_STEP, OMEGA_MIN, OMEGA_MAX)
        if keys[pygame.K_RIGHT]:
            omega_cmd = clamp(omega_cmd - OMEGA_STEP, OMEGA_MIN, OMEGA_MAX)
        if not keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            omega_cmd *= 0.85
            if abs(omega_cmd) < 1e-3:
                omega_cmd = 0.0

        robot.apply(v_cmd, omega_cmd, DT)
        robot.check_collision(obstacles)
        trail.append((robot.x, robot.y))
        if len(trail) > 2000:
            trail.pop(0)

        screen.fill(C_BG)
        draw_grid(screen)
        for o in obstacles:
            o.draw(screen)
        draw_goal(screen)
        draw_trail(screen, trail)
        robot.draw(screen)

        lines = [
            f"v = {v_cmd:5.2f} m/s   omega = {omega_cmd:6.2f} rad/s",
            f"pos = ({robot.x:5.2f}, {robot.y:5.2f})   angle = {math.degrees(robot.theta):6.1f} deg",
            "COLLISION!" if robot.collided else "clear",
            "UP/DOWN: speed   LEFT/RIGHT: turn   SPACE: stop   R: reset   ESC: quit",
        ]
        for i, line in enumerate(lines):
            color = (220, 40, 40) if line == "COLLISION!" else C_TEXT
            screen.blit(font.render(line, True, color), (8, 8 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()