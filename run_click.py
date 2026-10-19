import sys
import math
import pygame

from config import (WINDOW_PX, FPS, DT, ARENA_M, START_POSE,
                    C_BG, C_GRID, C_GOAL, C_TRAIL, C_PRED, C_TEXT)
from geometry import world_to_screen, screen_to_world
from obstacles import default_obstacle_field
from robot import Robot
from controller import MPCController


def draw_grid(surface):
    for m in range(int(ARENA_M) + 1):
        x0, y0 = world_to_screen(m, 0)
        x1, y1 = world_to_screen(m, ARENA_M)
        pygame.draw.line(surface, C_GRID, (x0, y0), (x1, y1), 1)
        hx0, hy0 = world_to_screen(0, m)
        hx1, hy1 = world_to_screen(ARENA_M, m)
        pygame.draw.line(surface, C_GRID, (hx0, hy0), (hx1, hy1), 1)


def draw_goal(surface, goal):
    if goal is None:
        return
    gx, gy = world_to_screen(*goal)
    s = 10
    pygame.draw.line(surface, C_GOAL, (gx - s, gy - s), (gx + s, gy + s), 3)
    pygame.draw.line(surface, C_GOAL, (gx - s, gy + s), (gx + s, gy - s), 3)


def draw_trail(surface, trail):
    if len(trail) > 1:
        pts = [world_to_screen(x, y) for (x, y) in trail]
        pygame.draw.lines(surface, C_TRAIL, False, pts, 2)


def draw_prediction(surface, X_pred):
    if X_pred is None:
        return
    pts = [world_to_screen(X_pred[0, k], X_pred[1, k])
           for k in range(X_pred.shape[1])]
    if len(pts) > 1:
        pygame.draw.lines(surface, C_PRED, False, pts, 2)
    for p in pts:
        pygame.draw.circle(surface, C_PRED, p, 3)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PX, WINDOW_PX))
    pygame.display.set_caption("run_click.py")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    obstacles = default_obstacle_field()
    robot = Robot(*START_POSE)
    goal = None
    controller = MPCController((robot.x, robot.y), obstacles)
    trail = []
    stuck_frames = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    robot = Robot(*START_POSE)
                    trail = []
                    goal = None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gx, gy = screen_to_world(*event.pos)
                inside = any(o.signed_distance(gx, gy, math) < 0 for o in obstacles)
                if not inside and 0 <= gx <= ARENA_M and 0 <= gy <= ARENA_M:
                    goal = (gx, gy)
                    stuck_frames = 0

        if goal is not None:
            dist = math.hypot(robot.x - goal[0], robot.y - goal[1])
            if dist > 0.3:
                v, omega, X_pred = controller.solve(robot.state, goal)
                robot.apply(v, omega, DT)
                robot.check_collision(obstacles)
                trail.append((robot.x, robot.y))
                if len(trail) > 5000:
                    trail.pop(0)
                if abs(v) < 0.05:
                    stuck_frames += 1
                else:
                    stuck_frames = 0
                if stuck_frames > 2 * FPS:
                    goal = None
                    stuck_frames = 0
            else:
                v, omega, X_pred = 0.0, 0.0, None
                stuck_frames = 0
        else:
            v, omega, X_pred = 0.0, 0.0, None

        screen.fill(C_BG)
        draw_grid(screen)
        for o in obstacles:
            o.draw(screen)
        draw_goal(screen, goal)
        draw_trail(screen, trail)
        draw_prediction(screen, X_pred)
        robot.draw(screen)

        if goal is None:
            status = "click to set a goal"
        elif math.hypot(robot.x - goal[0], robot.y - goal[1]) < 0.3:
            status = "arrived - click again"
        else:
            status = "driving..."
        hud = [
            f"v = {v:5.2f} m/s   omega = {omega:6.2f} rad/s",
            f"pos = ({robot.x:5.2f}, {robot.y:5.2f})",
            status,
        ]
        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, C_TEXT), (8, 8 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()