"""Headless experiment harness for the reviewer's 4 questions.

Runs the *actual* project controller (controller.MPCController) in closed loop,
no pygame window. Produces three figures:

  Q3  fig_q3_no_avoidance.png   -- static scene, obstacle avoidance DISABLED
  Q4a fig_q4a_long_obstacle.png -- static scene, avoidance ON, recovery ON/OFF
  Q4b fig_q4b_fast_dynamic.png  -- crossing obstacles, slow vs comparable speed
"""
import os, sys, math, time
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

PROJ = "/home/sushanth/Documents/mpc_ws"
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DT, START_POSE, WAYPOINTS, GOAL_THRESHOLD, XLIM, YLIM,
                    V_MAX, OMEGA_MAX, ROBOT_RADIUS_M)
from obstacles import load_scene, Obstacle
from robot import Robot, unicycle_step
from controller import MPCController

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def min_clearance(x, y, obstacles, t):
    if not obstacles:
        return float("inf")
    return min(o.signed_distance(x, y, math, t) for o in obstacles)


def run_loop(waypoints, sim_obstacles, ctrl_obstacles, use_recovery,
             max_steps=1500):
    """Closed loop. sim_obstacles = ground truth for collision/clearance;
    ctrl_obstacles = what the MPC is told about (empty -> no avoidance)."""
    robot = Robot(*START_POSE)
    controller = MPCController(waypoints[0], ctrl_obstacles)
    sim_t = 0.0
    wp = 0
    log = dict(traj=[], ctrl=[], clr=[], solve=[], recov=[], t=[])
    stuck = 0
    recov = 0
    STUCK_SPEED, STUCK_STEPS, RECOVERY_STEPS = 0.05, 15, 12
    collisions = 0
    reached = []

    for step in range(max_steps):
        target = waypoints[wp]
        dist = math.hypot(robot.x - target[0], robot.y - target[1])

        if recov > 0:
            bearing = math.atan2(target[1] - robot.y, target[0] - robot.x)
            ang_err = math.atan2(math.sin(bearing - robot.theta),
                                 math.cos(bearing - robot.theta))
            omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * ang_err))
            v = 0.0
            if abs(ang_err) < 0.5:
                nx = robot.x + 0.3 * math.cos(robot.theta) * DT
                ny = robot.y + 0.3 * math.sin(robot.theta) * DT
                if all(not o.collides_circle(nx, ny, robot.r + 0.1, sim_t)
                       for o in sim_obstacles):
                    v = 0.3
            recov -= 1
            solve_dt = 0.0
            in_recov = True
        else:
            t0 = time.time()
            v, omega, _ = controller.solve(robot.state, target, t0=sim_t)
            solve_dt = time.time() - t0
            in_recov = False

        robot.apply(v, omega, DT)
        sim_t += DT
        hit = robot.check_collision(sim_obstacles, sim_t)
        if hit:
            collisions += 1

        clr = min_clearance(robot.x, robot.y, sim_obstacles, sim_t)
        log["traj"].append((robot.x, robot.y))
        log["ctrl"].append((v, omega))
        log["clr"].append(clr)
        log["solve"].append(solve_dt * 1000.0)
        log["recov"].append(in_recov)
        log["t"].append(sim_t)

        if use_recovery and not in_recov:
            if abs(v) < STUCK_SPEED and dist > GOAL_THRESHOLD:
                stuck += 1
                if stuck >= STUCK_STEPS:
                    recov = RECOVERY_STEPS
                    stuck = 0
            else:
                stuck = 0

        if dist < GOAL_THRESHOLD:
            reached.append(wp)
            if wp < len(waypoints) - 1:
                wp += 1
            else:
                break

    log = {k: np.array(v) if k not in ("traj", "ctrl") else np.array(v)
           for k, v in log.items()}
    log["collisions"] = collisions
    log["reached"] = len(set(reached))
    log["n_wp"] = len(waypoints)
    log["final_wp"] = wp
    log["steps"] = step + 1
    return log


def draw_obstacles(ax, obstacles, t=0.0):
    for o in obstacles:
        cs = o.corners(t) + [o.corners(t)[0]]
        xs = [c[0] for c in cs]; ys = [c[1] for c in cs]
        color = "navajowhite" if o.is_dynamic else "lightcoral"
        edge = "darkorange" if o.is_dynamic else "indianred"
        ax.fill(xs, ys, color=color, alpha=0.7, zorder=1)
        ax.plot(xs, ys, color=edge, lw=1, zorder=1)


# ---------------------------------------------------------------- Q3
def q3():
    print("Q3: no obstacle avoidance ...")
    obs = load_scene("static")
    # disable avoidance: tell the MPC only about one far-away dummy box, so the
    # collision constraint is present but trivially satisfied everywhere.
    noavoid = [Obstacle(1e4, 1e4, 0.01, 0.01)]
    log = run_loop(WAYPOINTS, sim_obstacles=obs, ctrl_obstacles=noavoid,
                   use_recovery=False)
    traj = log["traj"]; ctrl = log["ctrl"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    draw_obstacles(a1, obs)
    a1.plot(traj[:, 0], traj[:, 1], "b-", lw=2.2, label="ego path", zorder=3)
    a1.plot(START_POSE[0], START_POSE[1], "ko", ms=9, label="start")
    for wp in WAYPOINTS:
        a1.plot(wp[0], wp[1], "rx", ms=11, mew=2.5)
    a1.set_title("Q3  MPC with obstacle cost/constraint REMOVED\n"
                 "(pure goal-pose regulation — plows straight through boxes)")
    a1.set_xlim(XLIM); a1.set_ylim(YLIM); a1.set_aspect("equal")
    a1.set_xlabel("x [m]"); a1.set_ylabel("y [m]"); a1.grid(alpha=.3); a1.legend(loc="upper left")

    t = np.arange(len(ctrl)) * DT
    a2.plot(t, ctrl[:, 0], "b-", label="v [m/s]")
    a2.plot(t, ctrl[:, 1], "r-", label="ω [rad/s]")
    a2.axhline(V_MAX, color="b", ls="--", alpha=.4)
    a2.axhline(OMEGA_MAX, color="r", ls="--", alpha=.4)
    a2.axhline(-OMEGA_MAX, color="r", ls="--", alpha=.4)
    a2.set_title("Control inputs — smooth, saturates v=vmax between waypoints")
    a2.set_xlabel("time [s]"); a2.grid(alpha=.3); a2.legend()
    fig.tight_layout(); p = os.path.join(OUT, "fig_q3_no_avoidance.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"   reached {log['reached']}/{log['n_wp']} wp, "
          f"collisions(vs real boxes)={log['collisions']}, steps={log['steps']}")
    return p, log


# ---------------------------------------------------------------- Q4a
def q4a():
    """Controlled wall-length sweep. Ego must get from (7,1) to (7,13); a thin
    horizontal wall of growing length sits across the corridor at y=7. As the
    wall exceeds the horizon's lateral reach the MPC can no longer 'see' a way
    around and traps in the local minimum right in front of it."""
    print("Q4a: wall-length sweep (local-minimum onset) ...")
    start = (7.0, 1.0, math.pi / 2)
    goal = (7.0, 13.0, math.pi / 2)
    lengths = [2.0, 5.0, 9.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    outs = {}
    for ax, L in zip(axes, lengths):
        wall = [Obstacle(7.0, 7.0, L, 0.5, angle=0.0)]
        log = run_from(start=start, waypoints=[goal], sim_obstacles=wall,
                       ctrl_obstacles=wall, use_recovery=False, max_steps=400)
        outs[L] = log
        draw_obstacles(ax, wall)
        tr = log["traj"]
        ax.plot(tr[:, 0], tr[:, 1], "b-", lw=2, zorder=3, label="ego path")
        ax.plot(start[0], start[1], "ko", ms=8, label="start")
        ax.plot(goal[0], goal[1], "rx", ms=12, mew=2.5, label="goal")
        ax.plot(tr[-1, 0], tr[-1, 1], "b*", ms=16, zorder=4, label="final")
        final_d = math.hypot(tr[-1, 0] - goal[0], tr[-1, 1] - goal[1])
        verdict = "REACHED" if log["reached"] else f"STUCK ({final_d:.1f} m short)"
        ax.set_title(f"Q4a  wall length = {L:.0f} m\n{verdict}, "
                     f"steps={log['steps']}")
        ax.set_xlim(0, 14); ax.set_ylim(0, 14); ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Long-obstacle failure onset — same start/goal, growing wall "
                 "(N=30 horizon ≈ 9 m reach)", fontsize=12)
    fig.tight_layout(); p = os.path.join(OUT, "fig_q4a_long_obstacle.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    for L, log in outs.items():
        tr = log["traj"]
        fd = math.hypot(tr[-1, 0] - goal[0], tr[-1, 1] - goal[1])
        print(f"   L={L}m: reached={log['reached']}, final_dist={fd:.2f} m, "
              f"steps={log['steps']}")
    return p, outs


# ---------------------------------------------------------------- Q4b
def make_crossing_scene(obs_speed):
    """A box that crosses the ego's straight corridor from below, timed to
    arrive at the crossing point as the ego does. Ego runs (1,7)->(14,7)."""
    # ego reaches x~7 at t ~ 6/3 = 2 s; box starts at (7,1) going up at v.
    # start y so it hits y=7 near t=2s only for the fast case -> worst timing.
    return [Obstacle(7.0, 7.0 - obs_speed * 2.0, 1.4, 1.0,
                     angle=math.pi / 2, v=obs_speed)]


def run_from_predictor(start, goal, sim_obstacles, use_velocity, max_steps=800):
    """Closed loop with a choice of predictor.
    use_velocity=True : MPC is told the true (v, omega) -> exact future pose.
    use_velocity=False: MPC assumes each obstacle is STATIC at its currently
                        observed pose (re-observed every tick, zero predicted
                        drift) -- the common assumption without velocity est."""
    robot = Robot(*start)
    # controller is built with static stand-ins; we overwrite their params each
    # tick from the true obstacle state.
    ctrl_obs = [Obstacle(o.cx0, o.cy0, o.w, o.h, o.angle0, 0.0, 0.0)
                for o in sim_obstacles]
    controller = MPCController(goal, ctrl_obs)
    sim_t = 0.0
    log = dict(traj=[], clr=[], t=[])
    collisions = 0; reached = 0
    for step in range(max_steps):
        # refresh what the controller "sees"
        for i, o in enumerate(sim_obstacles):
            cx, cy, ang = o.pose_at(sim_t)
            if use_velocity:
                controller._obs_p[i] = [o.cx0, o.cy0, o.w / 2, o.h / 2,
                                        o.angle0, o._dx, o._dy, o.omega]
            else:
                controller._obs_p[i] = [cx, cy, o.w / 2, o.h / 2, ang,
                                        0.0, 0.0, 0.0]
        dist = math.hypot(robot.x - goal[0], robot.y - goal[1])
        v, omega, _ = controller.solve(robot.state, goal, t0=sim_t)
        robot.apply(v, omega, DT)
        sim_t += DT
        if robot.check_collision(sim_obstacles, sim_t):
            collisions += 1
        log["traj"].append((robot.x, robot.y))
        log["clr"].append(min_clearance(robot.x, robot.y, sim_obstacles, sim_t))
        log["t"].append(sim_t)
        if dist < GOAL_THRESHOLD:
            reached = 1; break
    for k in log:
        log[k] = np.array(log[k])
    log["collisions"] = collisions; log["reached"] = reached
    log["steps"] = step + 1
    return log


def q4b():
    """Obstacle-speed sweep, static-assumption predictor. The controller re-
    observes the obstacle each tick but assumes it is FROZEN over the horizon.
    As v_obs -> v_ego that assumption is wrong by a growing margin and the ego
    is clipped. Our shipped controller instead predicts the exact future pose
    (constant-velocity model), which handles all three speeds -- reported too."""
    print("Q4b: crossing-obstacle speed sweep (static-assumption predictor) ...")
    goal = (14.0, 7.0, 0.0)
    start = (1.0, 7.0, 0.0)
    speeds = [(0.3, "slow"), (1.5, "half ego speed"), (3.0, "= ego speed")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    results = {}
    for ax, (spd, label) in zip(axes, speeds):
        obs = make_crossing_scene(spd)
        naive = run_from_predictor(start, goal, obs, use_velocity=False)
        exact = run_from_predictor(start, goal, [make_crossing_scene(spd)[0]],
                                   use_velocity=True)
        results[spd] = (naive, exact)
        tend = naive["t"][-1]
        for tt in (0.0, tend * 0.4, tend * 0.7, tend):
            draw_obstacles(ax, obs, t=tt)
        tr = naive["traj"]
        ax.plot(tr[:, 0], tr[:, 1], "b-", lw=2, zorder=3, label="ego (static-assume)")
        ax.plot(start[0], start[1], "ko", ms=8, label="start")
        ax.plot(goal[0], goal[1], "rx", ms=11, mew=2.5, label="goal")
        mc = naive["clr"].min() - ROBOT_RADIUS_M
        hit = "COLLISION" if naive["collisions"] > 0 else "clear"
        ex_hit = "clears" if exact["collisions"] == 0 else "collides"
        ax.set_title(f"v_obs={spd} m/s ({label})\n"
                     f"static-assume: {hit}, min clr={mc:+.2f} m\n"
                     f"exact-velocity predictor: {ex_hit}")
        ax.set_xlim(-1, 15); ax.set_ylim(-1, 15); ax.set_aspect("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=.3)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Q4b  Dynamic obstacle, growing speed — failure comes from "
                 "PREDICTION ERROR, not speed itself\n(ghosts = obstacle at "
                 "t = 0, 0.4, 0.7, 1.0·T)", fontsize=12)
    fig.tight_layout(); p = os.path.join(OUT, "fig_q4b_fast_dynamic.png")
    fig.savefig(p, dpi=110); plt.close(fig)
    for spd, (naive, exact) in results.items():
        mc = naive["clr"].min() - ROBOT_RADIUS_M
        print(f"   v_obs={spd}: STATIC-assume collisions={naive['collisions']} "
              f"min_clr={mc:+.2f} m | EXACT-pred collisions={exact['collisions']}")
    return p, results


def run_from(start, waypoints, sim_obstacles, ctrl_obstacles, use_recovery,
             max_steps=1500):
    """Same as run_loop but with an explicit start pose."""
    robot = Robot(*start)
    controller = MPCController(waypoints[0], ctrl_obstacles)
    sim_t = 0.0
    wp = 0
    log = dict(traj=[], ctrl=[], clr=[], solve=[], t=[])
    stuck = 0; recov = 0
    STUCK_SPEED, STUCK_STEPS, RECOVERY_STEPS = 0.05, 15, 12
    collisions = 0; reached = 0
    for step in range(max_steps):
        target = waypoints[wp]
        dist = math.hypot(robot.x - target[0], robot.y - target[1])
        if recov > 0:
            bearing = math.atan2(target[1] - robot.y, target[0] - robot.x)
            ang_err = math.atan2(math.sin(bearing - robot.theta),
                                 math.cos(bearing - robot.theta))
            omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * ang_err))
            v = 0.0
            if abs(ang_err) < 0.5:
                nx = robot.x + 0.3 * math.cos(robot.theta) * DT
                ny = robot.y + 0.3 * math.sin(robot.theta) * DT
                if all(not o.collides_circle(nx, ny, robot.r + 0.1, sim_t)
                       for o in sim_obstacles):
                    v = 0.3
            recov -= 1
        else:
            v, omega, _ = controller.solve(robot.state, target, t0=sim_t)
        robot.apply(v, omega, DT)
        sim_t += DT
        if robot.check_collision(sim_obstacles, sim_t):
            collisions += 1
        clr = min_clearance(robot.x, robot.y, sim_obstacles, sim_t)
        log["traj"].append((robot.x, robot.y))
        log["ctrl"].append((v, omega))
        log["clr"].append(clr); log["t"].append(sim_t)
        if use_recovery:
            if abs(v) < STUCK_SPEED and dist > GOAL_THRESHOLD:
                stuck += 1
                if stuck >= STUCK_STEPS:
                    recov = RECOVERY_STEPS; stuck = 0
            else:
                stuck = 0
        if dist < GOAL_THRESHOLD:
            reached = 1
            break
    for k in ("traj", "ctrl", "clr", "t"):
        log[k] = np.array(log[k])
    log["collisions"] = collisions; log["reached"] = reached
    log["steps"] = step + 1
    return log


if __name__ == "__main__":
    p3, l3 = q3()
    p4a, outs4a = q4a()
    p4b, r4b = q4b()
    print("\nFIGURES:")
    for p in (p3, p4a, p4b):
        print("  ", p)
