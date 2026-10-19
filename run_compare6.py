
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import DT, ROBOT_RADIUS_M, GOAL_THRESHOLD, OMEGA_MAX
from obstacles import Obstacle
from robot import Robot
from controller import MPCController

START = (1.0, 7.0, 0.0)
GOAL = (14.0, 7.0, 0.0)
SPEEDS = [0.3, 1.5, 3.0]


def crossing_scene(v_obs):
    return [Obstacle(7.0, 7.0 - v_obs * 2.0, 1.4, 1.0, angle=math.pi / 2, v=v_obs)]


def refresh_params(controller, obstacles, t, use_velocity):
    for i, o in enumerate(obstacles):
        if use_velocity:
            controller._obs_p[i] = [o.cx0, o.cy0, o.w / 2, o.h / 2, o.angle0,
                                    o._dx, o._dy, o.omega]
        else:
            cx, cy, ang = o.pose_at(t)
            controller._obs_p[i] = [cx, cy, o.w / 2, o.h / 2, ang, 0.0, 0.0, 0.0]


def run_case(use_velocity, v_obs, max_steps=140):
    obs = crossing_scene(v_obs)
    stand_ins = [Obstacle(o.cx0, o.cy0, o.w, o.h, o.angle0, 0, 0) for o in obs]
    controller = MPCController(GOAL, stand_ins)
    robot = Robot(*START)
    t = 0.0
    traj, preds, times = [], [], []
    collided = False
    min_clr = float("inf")
    for _ in range(max_steps):
        refresh_params(controller, obs, t, use_velocity)
        v, om, X = controller.solve(robot.state, GOAL, t0=t)
        robot.apply(v, om, DT)
        t += DT
        if robot.check_collision(obs, t):
            collided = True
        min_clr = min(min_clr, min(o.signed_distance(robot.x, robot.y, math, t)
                                   for o in obs) - ROBOT_RADIUS_M)
        traj.append((robot.x, robot.y))
        preds.append(None if X is None else np.array(X))
        times.append(t)
        if math.hypot(robot.x - GOAL[0], robot.y - GOAL[1]) < GOAL_THRESHOLD:
            break
    return {"obs": obs, "traj": np.array(traj), "preds": preds,
            "times": times, "collided": collided, "min_clr": min_clr,
            "v": v_obs, "exact": use_velocity, "n": len(traj)}


def draw_obstacles(ax, obstacles, t):
    for o in obstacles:
        cs = o.corners(t) + [o.corners(t)[0]]
        ax.fill([c[0] for c in cs], [c[1] for c in cs],
                color="navajowhite", alpha=0.8, zorder=1)
        ax.plot([c[0] for c in cs], [c[1] for c in cs],
                color="darkorange", lw=1, zorder=1)


def main():
    print("computing 6 runs ...")
    cases = [[run_case(True, v) for v in SPEEDS],
             [run_case(False, v) for v in SPEEDS]]
    for row in cases:
        for c in row:
            print(f"  {'EXACT ' if c['exact'] else 'STATIC'} v={c['v']}: "
                  f"{'COLLISION' if c['collided'] else 'cleared'}, "
                  f"min_clr={c['min_clr']:+.2f} m")
    n_frames = max(c["n"] for row in cases for c in row)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle("Crossing obstacle — EXACT (top) vs STATIC assumption (bottom)",
                 fontsize=13)

    def draw_frame(fi):
        for r in range(2):
            for cix in range(3):
                ax = axes[r][cix]
                c = cases[r][cix]
                i = min(fi, c["n"] - 1)
                t = c["times"][i]
                ax.clear()
                ax.set_xlim(-1, 15); ax.set_ylim(-1, 15); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                draw_obstacles(ax, c["obs"], t)
                tr = c["traj"]
                ax.plot(tr[:i + 1, 0], tr[:i + 1, 1], "b-", lw=2)
                p = c["preds"][i]
                if p is not None:
                    ax.plot(p[0, :], p[1, :], "g.-", ms=2, lw=0.8)
                ax.plot(tr[i, 0], tr[i, 1], "o", color="royalblue", ms=9)
                ax.plot(START[0], START[1], "ko", ms=5)
                ax.plot(GOAL[0], GOAL[1], "rx", ms=9, mew=2)
                status = "COLLISION" if (c["collided"] and i >= c["n"] - 1) \
                    else ("cleared" if i >= c["n"] - 1 else "")
                col = "crimson" if c["collided"] else "green"
                mode = "EXACT" if c["exact"] else "STATIC"
                ax.set_title(f"{mode}   v_obs={c['v']} m/s   {status}",
                             color=(col if status else "black"), fontsize=10)
        return []

    anim = animation.FuncAnimation(fig, draw_frame, frames=n_frames,
                                   interval=70, blit=False, repeat=True)
    fig.tight_layout()
    plt.show()
    return anim


if __name__ == "__main__":
    main()
