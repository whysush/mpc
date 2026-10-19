"""Live demo for Q4b: a dynamic obstacle crossing the ego's path.

Watch how the MPC handles a box that crosses its corridor at a chosen speed,
and compare the two prediction models:

    python run_crossing.py                 # v_obs = 3.0 m/s, exact predictor
    python run_crossing.py 3.0 exact       # predicts obstacle's future pose
    python run_crossing.py 3.0 static      # assumes obstacle is stationary
    python run_crossing.py 1.5 static      # half ego speed, static assumption

  exact  : the controller propagates the obstacle forward using its known
           velocity (this is what the delivered controller does) -> clears it.
  static : the controller re-sees the obstacle each tick but assumes it is
           frozen over the horizon (the common case without velocity estimates)
           -> collides once v_obs is comparable to the ego speed.

Keys:  R = reset,  E = exact predictor,  S = static predictor,
       UP/DOWN = change obstacle speed,  ESC = quit
"""
import sys
import math
import pygame

from config import (WINDOW_PX, FPS, DT, XLIM, YLIM, GOAL_THRESHOLD,
                    C_BG, C_TEXT, ROBOT_RADIUS_M)
from obstacles import Obstacle
from robot import Robot
from controller import MPCController
from run_mpc import draw_grid, draw_trail, draw_prediction

GOAL = (14.0, 7.0, 0.0)
START = (1.0, 7.0, 0.0)


def make_scene(v_obs):
    """A box that starts below the corridor and crosses upward, timed to reach
    the ego's line (y=7) at about the moment the ego passes x=7."""
    return [Obstacle(7.0, 7.0 - v_obs * 2.0, 1.4, 1.0,
                     angle=math.pi / 2, v=v_obs)]


def obs_params(o, t, use_velocity):
    """Parameter row the controller uses for this obstacle."""
    if use_velocity:
        return [o.cx0, o.cy0, o.w / 2, o.h / 2, o.angle0, o._dx, o._dy, o.omega]
    cx, cy, ang = o.pose_at(t)          # re-observe current pose, assume frozen
    return [cx, cy, o.w / 2, o.h / 2, ang, 0.0, 0.0, 0.0]


def build(v_obs):
    obs = make_scene(v_obs)
    # controller is built with static stand-ins; params refreshed every tick.
    ctrl_obs = [Obstacle(o.cx0, o.cy0, o.w, o.h, o.angle0, 0.0, 0.0)
                for o in obs]
    return obs, MPCController(GOAL, ctrl_obs)


def main():
    v_obs = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    use_velocity = not (len(sys.argv) > 2 and sys.argv[2].lower() == "static")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PX, WINDOW_PX))
    pygame.display.set_caption("Q4b - crossing dynamic obstacle")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    obs, controller = build(v_obs)
    robot = Robot(*START)
    trail = []
    sim_t = 0.0
    ever_hit = False
    min_clr = float("inf")
    arrived = False

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key in (pygame.K_r, pygame.K_e, pygame.K_s,
                               pygame.K_UP, pygame.K_DOWN):
                    if e.key == pygame.K_e:
                        use_velocity = True
                    elif e.key == pygame.K_s:
                        use_velocity = False
                    elif e.key == pygame.K_UP:
                        v_obs = round(v_obs + 0.5, 1)
                    elif e.key == pygame.K_DOWN:
                        v_obs = max(0.0, round(v_obs - 0.5, 1))
                    obs, controller = build(v_obs)
                    robot = Robot(*START)
                    trail = []
                    sim_t = 0.0
                    ever_hit = False
                    min_clr = float("inf")
                    arrived = False

        X_pred = None
        if not arrived:
            for i, o in enumerate(obs):
                controller._obs_p[i] = obs_params(o, sim_t, use_velocity)
            v, omega, X_pred = controller.solve(robot.state, GOAL, t0=sim_t)
            robot.apply(v, omega, DT)
            sim_t += DT
            if robot.check_collision(obs, sim_t):
                ever_hit = True
            clr = min(o.signed_distance(robot.x, robot.y, math, sim_t)
                      for o in obs) - ROBOT_RADIUS_M
            min_clr = min(min_clr, clr)
            trail.append((robot.x, robot.y))
            if math.hypot(robot.x - GOAL[0], robot.y - GOAL[1]) < GOAL_THRESHOLD:
                arrived = True
        else:
            v = omega = 0.0

        screen.fill(C_BG)
        draw_grid(screen)
        for o in obs:
            o.draw(screen, sim_t)
            o.draw_parameterization(screen, sim_t)
        from geometry import world_to_screen
        gx, gy = world_to_screen(GOAL[0], GOAL[1])
        pygame.draw.line(screen, (220, 30, 30), (gx - 10, gy - 10),
                         (gx + 10, gy + 10), 3)
        pygame.draw.line(screen, (220, 30, 30), (gx - 10, gy + 10),
                         (gx + 10, gy - 10), 3)
        draw_trail(screen, trail)
        draw_prediction(screen, X_pred)
        robot.draw(screen)

        mode = "EXACT (predicts future pose)" if use_velocity \
            else "STATIC (assumes frozen)"
        hud = [
            f"predictor = {mode}",
            f"v_obs = {v_obs:.1f} m/s   (ego v_max = 3.0)",
            f"min clearance so far = {min_clr:+.2f} m",
            "COLLISION!" if ever_hit else ("ARRIVED" if arrived else "clear"),
            "E: exact   S: static   UP/DOWN: v_obs   R: reset   ESC: quit",
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
