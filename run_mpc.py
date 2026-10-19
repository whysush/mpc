import sys
import time
import math
import pygame
import numpy as np

from config import (WINDOW_PX, FPS, DT, ARENA_M, XLIM, YLIM, START_POSE,
                    WAYPOINTS, GOAL_THRESHOLD, SHOW_PREDICTIONS,
                    SHOW_OBSTACLE_PARAMETERIZATION,
                    C_BG, C_GRID, C_GOAL, C_TRAIL, C_PRED, C_TEXT,
                    V_MAX, OMEGA_MAX)
from geometry import world_to_screen
from obstacles import load_scene
from robot import Robot
from controller import MPCController
from planner import GlobalPlanner, LOOKAHEAD

# global-path overlay colors
C_PATH = (150, 40, 200)     # A* route (purple)
C_CARROT = (255, 140, 0)    # carrot the MPC chases (orange)
REPLAN_EVERY = 5            # dynamic scenes: replan A* every N steps


def draw_grid(surface):
    x0i, x1i = int(math.floor(XLIM[0])), int(math.ceil(XLIM[1]))
    y0i, y1i = int(math.floor(YLIM[0])), int(math.ceil(YLIM[1]))
    for m in range(x0i, x1i + 1):
        pygame.draw.line(surface, C_GRID,
                         world_to_screen(m, YLIM[0]),
                         world_to_screen(m, YLIM[1]), 1)
    for m in range(y0i, y1i + 1):
        pygame.draw.line(surface, C_GRID,
                         world_to_screen(XLIM[0], m),
                         world_to_screen(XLIM[1], m), 1)


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


def draw_path(surface, path):
    """Draw the global A* route and its vertices."""
    if not path or len(path) < 2:
        return
    pts = [world_to_screen(x, y) for (x, y) in path]
    pygame.draw.lines(surface, C_PATH, False, pts, 2)
    for p in pts:
        pygame.draw.circle(surface, C_PATH, p, 3)


def draw_carrot(surface, carrot):
    if carrot is None:
        return
    pygame.draw.circle(surface, C_CARROT, world_to_screen(carrot[0], carrot[1]),
                       6, 2)


def draw_waypoint_marker(surface, wp, active):
    wx, wy = world_to_screen(wp[0], wp[1])
    if active:
        s = 10
        pygame.draw.line(surface, C_GOAL, (wx - s, wy - s), (wx + s, wy + s), 3)
        pygame.draw.line(surface, C_GOAL, (wx - s, wy + s), (wx + s, wy - s), 3)
        if len(wp) > 2:  # show target heading
            hx = wp[0] + 0.6 * math.cos(wp[2])
            hy = wp[1] + 0.6 * math.sin(wp[2])
            pygame.draw.line(surface, C_GOAL, (wx, wy),
                             world_to_screen(hx, hy), 2)
    else:
        pygame.draw.circle(surface, (150, 150, 150), (wx, wy), 5, 2)


def save_plots(traj, controls, solve_times, scene):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots. (pip install matplotlib)")
        return

    traj = np.array(traj)
    controls = np.array(controls)
    solve_times = np.array(solve_times) * 1000.0  # ms

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    obstacles = load_scene(scene)
    for o in obstacles:
        cs = o.corners() + [o.corners()[0]]
        xs = [c[0] for c in cs]; ys = [c[1] for c in cs]
        color = "navajowhite" if o.is_dynamic else "lightcoral"
        edge = "darkorange" if o.is_dynamic else "indianred"
        ax1.fill(xs, ys, color=color, alpha=0.7)
        ax1.plot(xs, ys, color=edge, linewidth=1)
    ax1.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2, label="trajectory")
    ax1.plot(START_POSE[0], START_POSE[1], "ko", markersize=10, label="start")
    for wp in WAYPOINTS:
        ax1.plot(wp[0], wp[1], "rx", markersize=10)
    ax1.set_title(f"Trajectory ({scene})")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_xlim(XLIM); ax1.set_ylim(YLIM); ax1.set_aspect("equal")
    ax1.grid(True); ax1.legend()

    t = np.arange(len(controls)) * DT
    ax2.plot(t, controls[:, 0], "b-", label="v [m/s]")
    ax2.plot(t, controls[:, 1], "r-", label="omega [rad/s]")
    ax2.axhline(V_MAX, color="b", ls="--", alpha=0.4)
    ax2.axhline(OMEGA_MAX, color="r", ls="--", alpha=0.4)
    ax2.axhline(-OMEGA_MAX, color="r", ls="--", alpha=0.4)
    ax2.set_title("Control inputs"); ax2.set_xlabel("time [s]"); ax2.set_ylabel("control")
    ax2.grid(True); ax2.legend()

    ax3.plot(solve_times, "k-", linewidth=0.8)
    ax3.fill_between(range(len(solve_times)), solve_times, alpha=0.3)
    ax3.axhline(solve_times.mean(), color="r", ls="--",
                label=f"mean: {solve_times.mean():.1f} ms")
    ax3.set_title("Solver performance"); ax3.set_xlabel("MPC step"); ax3.set_ylabel("solve time [ms]")
    ax3.grid(True); ax3.legend()

    plt.tight_layout()
    out = f"mpc_performance_{scene}.png"
    plt.savefig(out, dpi=100)
    print(f"Saved performance plots to {out}")


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "static"

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_PX, WINDOW_PX))
    pygame.display.set_caption(f"MPC - scene: {scene}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    obstacles = load_scene(scene)
    robot = Robot(*START_POSE)
    controller = MPCController(WAYPOINTS[0], obstacles)

    # global A* layer: routes around obstacles so the local MPC can't stall in
    # a local minimum. Static scenes plan once per segment; dynamic scenes
    # replan periodically on the obstacles' current pose.
    planner = GlobalPlanner(obstacles)
    dynamic_scene = any(o.is_dynamic for o in obstacles)
    planner.plan(START_POSE[:2], WAYPOINTS[0][:2], t=0.0)
    step_count = 0

    trail = []
    wp_index = 0
    sim_t = 0.0  # absolute sim clock, drives dynamic obstacle motion

    traj_log, control_log, time_log = [], [], []
    arrived = False

    # stuck recovery - backs up by a lil and rotates in place to find a better route
    STUCK_SPEED = 0.05      
    STUCK_STEPS = 15       
    RECOVERY_STEPS = 12     
    stuck_count = 0
    recovery_count = 0

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
                    traj_log, control_log, time_log = [], [], []
                    arrived = False
                    wp_index = 0
                    sim_t = 0.0
                    step_count = 0
                    planner.plan(START_POSE[:2], WAYPOINTS[0][:2], t=0.0)

        target = WAYPOINTS[wp_index]
        dist_to_goal = math.hypot(robot.x - target[0], robot.y - target[1])

        # --- global A* layer: (re)plan and pick the carrot to chase ---
        # Dynamic scenes replan on the obstacles' current pose every few steps;
        # a lost path (planner returned None) is retried immediately.
        if not arrived:
            if planner.path is None or (dynamic_scene
                                        and step_count % REPLAN_EVERY == 0):
                planner.plan((robot.x, robot.y), target[:2], t=sim_t)
        carrot = planner.carrot((robot.x, robot.y))
        # On final approach (or if A* found nothing) chase the true waypoint
        # pose so the goal heading is honored; otherwise chase the carrot.
        if carrot is None or dist_to_goal <= LOOKAHEAD:
            mpc_goal = target
        else:
            mpc_goal = carrot

        if not arrived:
            if recovery_count > 0:
                # --- recovery burst: rotate toward the carrot, creep forward ---
                aim = carrot if carrot is not None else target
                bearing = math.atan2(aim[1] - robot.y, aim[0] - robot.x)
                ang_err = math.atan2(math.sin(bearing - robot.theta),
                                     math.cos(bearing - robot.theta))
                omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * ang_err))
                # small forward creep only if roughly facing the goal AND the
                # next step stays clear, so recovery never drives into a box.
                v = 0.0
                if abs(ang_err) < 0.5:
                    nx = robot.x + 0.3 * math.cos(robot.theta) * DT
                    ny = robot.y + 0.3 * math.sin(robot.theta) * DT
                    if all(not o.collides_circle(nx, ny, robot.r + 0.1, sim_t)
                           for o in obstacles):
                        v = 0.3
                X_pred = None
                recovery_count -= 1
                solve_dt = 0.0
            else:
                # MPC tracks the A* carrot (or the true waypoint on final
                # approach); the receding-horizon optimization still does the
                # fine kinodynamic + dynamic-obstacle avoidance.
                t0 = time.time()
                v, omega, X_pred = controller.solve(robot.state, mpc_goal,
                                                    t0=sim_t)
                solve_dt = time.time() - t0

            robot.apply(v, omega, DT)
            sim_t += DT
            step_count += 1
            robot.check_collision(obstacles, sim_t)

            # detect a stall and trigger recovery
            if recovery_count == 0:
                if abs(v) < STUCK_SPEED and dist_to_goal > GOAL_THRESHOLD:
                    stuck_count += 1
                    if stuck_count >= STUCK_STEPS:
                        recovery_count = RECOVERY_STEPS
                        stuck_count = 0
                else:
                    stuck_count = 0

            traj_log.append((robot.x, robot.y))
            control_log.append((v, omega))
            time_log.append(solve_dt)
            trail.append((robot.x, robot.y))
            if len(trail) > 5000:
                trail.pop(0)

            if dist_to_goal < GOAL_THRESHOLD:
                if wp_index < len(WAYPOINTS) - 1:
                    wp_index += 1
                    # plan the next segment right away
                    planner.plan((robot.x, robot.y),
                                 WAYPOINTS[wp_index][:2], t=sim_t)
                else:
                    arrived = True
                    save_plots(traj_log, control_log, time_log, scene)
        else:
            v, omega, X_pred = 0.0, 0.0, None

        screen.fill(C_BG)
        draw_grid(screen)
        for o in obstacles:
            o.draw(screen, sim_t)
        if SHOW_OBSTACLE_PARAMETERIZATION:
            for o in obstacles:
                o.draw_parameterization(screen, sim_t)
        for j, wp in enumerate(WAYPOINTS):
            draw_waypoint_marker(screen, wp, j == wp_index and not arrived)
        draw_path(screen, planner.path)
        if not arrived:
            draw_carrot(screen, carrot)
        draw_trail(screen, trail)
        if SHOW_PREDICTIONS:
            draw_prediction(screen, X_pred)
        robot.draw(screen)

        avg_ms = (sum(time_log) / len(time_log) * 1000) if time_log else 0.0
        hud = [
            f"scene = {scene}   t = {sim_t:5.1f} s",
            f"v = {v:5.2f} m/s   omega = {omega:6.2f} rad/s",
            f"pos = ({robot.x:5.2f}, {robot.y:5.2f})   dist = {dist_to_goal:4.2f} m",
            f"avg solve = {avg_ms:5.1f} ms",
            f"waypoint {wp_index + 1} of {len(WAYPOINTS)}" if not arrived else "TOUR DONE (plots saved)",
            "COLLISION!" if robot.collided else "clear",
            "R: reset   ESC: quit",
        ]
        for i, line in enumerate(hud):
            color = (220, 40, 40) if line == "COLLISION!" else C_TEXT
            screen.blit(font.render(line, True, color), (8, 8 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
