"""Live demo: stationary obstacles + fast-moving obstacles (comparable speed).

The ego crosses a corridor left -> right while several boxes cross its path at a
speed close to its own top speed (ego v_max = 3.0 m/s). The MPC predicts each
moving obstacle's future pose, so it threads between them.

    python3 run_mixed.py            # movers at 2.5 m/s (default)
    python3 run_mixed.py 3.0        # movers at ego speed
    python3 run_mixed.py 1.5        # movers at half ego speed

Keys:  R = reset,  UP/DOWN = change mover speed,  ESC = quit
"""
import sys
import math
import pygame

from config import (WINDOW_PX, FPS, DT, XLIM, YLIM, GOAL_THRESHOLD,
                    C_BG, C_TEXT, OMEGA_MAX)
from obstacles import Obstacle
from robot import Robot
from controller import MPCController
from geometry import world_to_screen
from run_mpc import draw_grid, draw_trail, draw_prediction

START = (0.5, 7.5, 0.0)
GOAL = (14.0, 7.5, 0.0)


def build_scene(v_obs):
    """3 stationary boxes + 3 boxes crossing the corridor at v_obs, timed to
    intersect the ego's path at roughly x = 4, 8, 12."""
    static = [
        Obstacle(5.0, 5.0, 1.5, 1.5, 0.3),
        Obstacle(9.0, 10.0, 1.8, 1.0, -0.2),
        Obstacle(11.0, 5.5, 1.2, 1.2, 0.0),
    ]
    movers = [
        Obstacle(4.0,  4.0, 1.0, 1.0, angle=math.pi / 2,  v=v_obs),   # up
        Obstacle(8.0, 15.0, 1.0, 1.0, angle=-math.pi / 2, v=v_obs),   # down
        Obstacle(12.0, -4.0, 1.0, 1.0, angle=math.pi / 2, v=v_obs),   # up
    ]
    return static + movers


def main():
    v_obs = float(sys.argv[1]) if len(sys.argv) > 1 else 2.5

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PX, WINDOW_PX))
    pygame.display.set_caption("MPC - stationary + fast-moving obstacles")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    def reset():
        obs = build_scene(v_obs)
        return obs, MPCController(GOAL, obs), Robot(*START), [], 0.0, False, float("inf"), False

    obs, controller, robot, trail, sim_t, ever_hit, min_clr, arrived = reset()
    stuck = recov = 0
    STUCK_SPEED, STUCK_STEPS, RECOVERY_STEPS = 0.05, 15, 12

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    obs, controller, robot, trail, sim_t, ever_hit, min_clr, arrived = reset()
                    stuck = recov = 0
                elif e.key in (pygame.K_UP, pygame.K_DOWN):
                    v_obs = max(0.0, round(v_obs + (0.5 if e.key == pygame.K_UP else -0.5), 1))
                    obs, controller, robot, trail, sim_t, ever_hit, min_clr, arrived = reset()
                    stuck = recov = 0

        X_pred = None
        if not arrived:
            dist = math.hypot(robot.x - GOAL[0], robot.y - GOAL[1])
            if recov > 0:
                bearing = math.atan2(GOAL[1] - robot.y, GOAL[0] - robot.x)
                ang_err = math.atan2(math.sin(bearing - robot.theta),
                                     math.cos(bearing - robot.theta))
                omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * ang_err))
                v = 0.0
                if abs(ang_err) < 0.5:
                    nx = robot.x + 0.3 * math.cos(robot.theta) * DT
                    ny = robot.y + 0.3 * math.sin(robot.theta) * DT
                    if all(not o.collides_circle(nx, ny, robot.r + 0.1, sim_t)
                           for o in obs):
                        v = 0.3
                recov -= 1
            else:
                v, omega, X_pred = controller.solve(robot.state, GOAL, t0=sim_t)

            robot.apply(v, omega, DT)
            sim_t += DT
            if robot.check_collision(obs, sim_t):
                ever_hit = True
            clr = min(o.signed_distance(robot.x, robot.y, math, sim_t)
                      for o in obs) - robot.r
            min_clr = min(min_clr, clr)
            trail.append((robot.x, robot.y))

            if recov == 0:
                if abs(v) < STUCK_SPEED and dist > GOAL_THRESHOLD:
                    stuck += 1
                    if stuck >= STUCK_STEPS:
                        recov = RECOVERY_STEPS
                        stuck = 0
                else:
                    stuck = 0
            if dist < GOAL_THRESHOLD:
                arrived = True
        else:
            v = omega = 0.0

        screen.fill(C_BG)
        draw_grid(screen)
        for o in obs:
            o.draw(screen, sim_t)
            o.draw_parameterization(screen, sim_t)
        gx, gy = world_to_screen(GOAL[0], GOAL[1])
        pygame.draw.line(screen, (220, 30, 30), (gx - 10, gy - 10), (gx + 10, gy + 10), 3)
        pygame.draw.line(screen, (220, 30, 30), (gx - 10, gy + 10), (gx + 10, gy - 10), 3)
        draw_trail(screen, trail)
        draw_prediction(screen, X_pred)
        robot.draw(screen)

        hud = [
            f"stationary boxes = 3   moving boxes = 3",
            f"mover speed = {v_obs:.1f} m/s   (ego v_max = 3.0)",
            f"v = {v:5.2f}   omega = {omega:6.2f}",
            f"min clearance so far = {min_clr:+.2f} m",
            "COLLISION!" if ever_hit else ("ARRIVED" if arrived else "clear"),
            "UP/DOWN: mover speed   R: reset   ESC: quit",
        ]
        for i, line in enumerate(hud):
            col = (220, 40, 40) if line == "COLLISION!" else C_TEXT
            screen.blit(font.render(line, True, col), (8, 8 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
