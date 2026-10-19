import math, sys, os
sys.path.insert(0, "/home/sushanth/Documents/mpc_ws")
os.chdir("/home/sushanth/Documents/mpc_ws")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle

OUT = "report_figures"
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

# ---------------- Fig 1: closed-loop architecture ----------------
def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=fc, ec="#333", lw=1.4))
    ax.text(x + w/2, y + h - 0.28, title, ha="center", va="top", fontsize=12, weight="bold")
    ax.text(x + w/2, y + h/2 - 0.22, body, ha="center", va="center", fontsize=9.5, linespacing=1.35)

def arrow(ax, p, q, label="", off=(0, 0.18), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.5,
                                 color="#333", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((p[0]+q[0])/2 + off[0], (p[1]+q[1])/2 + off[1], label, ha="center",
                fontsize=9.5, style="italic", color="#222")

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
box(ax, 0.3, 3.6, 2.8, 2.2, "Mission", "config.yaml\nstart pose + 4 waypoints\n(x, y, θ)\ngoal threshold 0.5 m", "#e8eefc")
box(ax, 4.2, 3.3, 4.4, 2.6, "NMPC  (CasADi + IPOPT)",
    "horizon N = 30, dt = 0.1 s\nmin  Σ‖p−p_goal‖²Q + ‖u‖²R + ‖Δu‖²\n"
    "s.t. unicycle dynamics,  |v|,|ω|, accel bounds\n      SDF(x_k, obs_i(t0+k·dt)) ≥ r + margin − s\n"
    "K = 10 nearest obstacles, warm start", "#fff2d6")
box(ax, 9.7, 3.6, 3.0, 2.2, "Robot (unicycle)", "x⁺ = x + v cosθ·dt\ny⁺ = y + v sinθ·dt\nθ⁺ = θ + ω·dt\nr = 0.4 m", "#e3f4e3")
box(ax, 4.2, 0.35, 4.4, 2.0, "Environment", "13 static / 20 moving + rotating\noriented boxes (.obs scenes)\nsigned-distance collision check", "#f5e3e3")
box(ax, 9.7, 0.45, 3.0, 1.7, "Stuck recovery", "|v| < 0.05 for 15 steps\n→ rotate to goal bearing,\nsafe creep (12 steps)", "#ece3f5")
box(ax, 0.3, 0.55, 2.8, 1.6, "Logging / Viz", "Pygame render, predicted\nhorizon, perf. plots", "#eeeeee")
arrow(ax, (3.1, 4.7), (4.2, 4.7), "active waypoint")
arrow(ax, (8.6, 4.9), (9.7, 4.9), "u₀* = (v, ω)")
arrow(ax, (9.7, 4.2), (8.6, 4.2), "state x(t)", off=(0, -0.38))
arrow(ax, (6.4, 2.35), (6.4, 3.3), "obstacle poses + velocities", off=(1.45, -0.05))
arrow(ax, (10.6, 3.6), (10.6, 2.15), "stall", off=(-0.4, 0))
arrow(ax, (11.8, 2.15), (11.8, 3.6), "override u", off=(0.62, 0))
arrow(ax, (4.2, 3.6), (3.1, 1.9), "trajectories,\nsolve times", off=(-0.75, 0.0))
ax.text(6.5, 6.05, "Closed-loop receding-horizon control: re-solve every 0.1 s, apply only the first control",
        ha="center", fontsize=12.5, weight="bold")
fig.savefig(f"{OUT}/fig1_system_architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: model + SDF constraint ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.35]})
# unicycle
th = math.radians(35); x0, y0, r = 0.0, 0.0, 0.4
a.add_patch(Circle((x0, y0), r, fc="#cfe0fb", ec="#1f4ea8", lw=2))
a.annotate("", xy=(x0 + 1.3*math.cos(th), y0 + 1.3*math.sin(th)), xytext=(x0, y0),
           arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1f4ea8"))
a.text(x0 + 1.35*math.cos(th), y0 + 1.35*math.sin(th) + 0.05, "v", fontsize=14, color="#1f4ea8", weight="bold")
a.plot([x0, x0 + 1.0], [y0, y0], "k--", lw=1)
ang = np.linspace(0, th, 30); a.plot(0.7*np.cos(ang), 0.7*np.sin(ang), "k", lw=1)
a.text(0.78, 0.18, "θ", fontsize=14)
ang = np.linspace(th + 0.6, th + 2.2, 30)
a.plot(0.65*np.cos(ang), 0.65*np.sin(ang), color="#c0392b", lw=1.8)
a.annotate("", xy=(0.65*math.cos(th+2.25), 0.65*math.sin(th+2.25)),
           xytext=(0.65*math.cos(th+2.1), 0.65*math.sin(th+2.1)),
           arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8))
a.text(-0.55, 0.75, "ω", fontsize=14, color="#c0392b", weight="bold")
a.plot(x0, y0, "ko", ms=4); a.text(0.08, 0.08, "(x, y)", fontsize=10)
a.annotate("", xy=(0, -r), xytext=(0, 0), arrowprops=dict(arrowstyle="<->", lw=1))
a.text(-0.18, -0.28, "r", fontsize=11)
a.text(-1.6, -1.5, "State  X = [x, y, θ]ᵀ     Control  U = [v, ω]ᵀ\n"
       "x_{k+1} = x_k + v_k cos θ_k · dt\ny_{k+1} = y_k + v_k sin θ_k · dt\nθ_{k+1} = θ_k + ω_k · dt\n"
       "0 ≤ v ≤ 3 m/s,  |ω| ≤ 1 rad/s\n|Δv| ≤ 1.5·dt,  |Δω| ≤ 1.5·dt", fontsize=10.5, va="top", family="DejaVu Sans Mono")
a.set_xlim(-1.7, 2.2); a.set_ylim(-3.1, 1.7); a.set_aspect("equal"); a.axis("off")
a.set_title("(a) Unicycle (differential-drive) model", fontsize=13)

# SDF field around an oriented box
cx, cy, w, h, ang0 = 0.0, 0.0, 2.4, 1.0, math.radians(25)
gx, gy = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-2.4, 2.4, 320))
dx, dy = gx - cx, gy - cy; c, s = math.cos(-ang0), math.sin(-ang0)
lx, ly = dx*c - dy*s, dx*s + dy*c
ax_, ay_ = np.abs(lx) - w/2, np.abs(ly) - h/2
sd = np.sqrt(np.maximum(ax_, 0)**2 + np.maximum(ay_, 0)**2) + np.minimum(np.maximum(ax_, ay_), 0)
cf = b.contourf(gx, gy, sd, levels=np.linspace(-0.5, 2.5, 25), cmap="viridis_r", extend="both")
fig.colorbar(cf, ax=b, label="signed distance d [m]", shrink=0.85)
b.contour(gx, gy, sd, levels=[0.6], colors="white", linewidths=2, linestyles="--")
b.contour(gx, gy, sd, levels=[0.0], colors="k", linewidths=2)
corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
R = np.array([[math.cos(ang0), -math.sin(ang0)], [math.sin(ang0), math.cos(ang0)]])
b.add_patch(Polygon(corners @ R.T, fc="#f4a6a6", ec="k", lw=1.5))
b.text(0, 0, "obstacle", ha="center", va="center", fontsize=10)
b.text(0.35, 1.75, "d = r + margin = 0.6 m", color="white", fontsize=10, weight="bold")
b.set_aspect("equal"); b.set_xlabel("x [m]"); b.set_ylabel("y [m]")
b.set_title("(b) Signed-distance collision constraint\nd(x_k, y_k ; obs(t₀ + k·dt)) ≥ r + margin − s_k", fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_model_and_constraint.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("figs 1,2 done")


# ---------------- Fig 3: receding-horizon snapshots (dynamic scene) ----------------
from config import DT, START_POSE, WAYPOINTS, GOAL_THRESHOLD, OMEGA_MAX, XLIM, YLIM
from obstacles import load_scene
from robot import Robot
from controller import MPCController
scene = sys.argv[1] if len(sys.argv) > 1 else "dynamic"
obstacles = load_scene(scene)
robot = Robot(*START_POSE); ctrl = MPCController(WAYPOINTS[0], obstacles)
wp, t, stuck, rec = 0, 0.0, 0, 0
log = []  # (t, x, y, theta, X_pred, wp)
for step in range(900):
    tgt = WAYPOINTS[wp]; dist = math.hypot(robot.x - tgt[0], robot.y - tgt[1])
    if rec > 0:
        br = math.atan2(tgt[1]-robot.y, tgt[0]-robot.x)
        e = math.atan2(math.sin(br-robot.theta), math.cos(br-robot.theta))
        om = max(-OMEGA_MAX, min(OMEGA_MAX, 2*e)); v = 0.0
        if abs(e) < 0.5:
            nx, ny = robot.x + 0.03*math.cos(robot.theta), robot.y + 0.03*math.sin(robot.theta)
            if all(not o.collides_circle(nx, ny, robot.r+0.1, t) for o in obstacles): v = 0.3
        X = None; rec -= 1
    else:
        v, om, X = ctrl.solve(robot.state, tgt, t0=t)
    log.append((t, robot.x, robot.y, robot.theta, None if X is None else np.array(X), wp))
    robot.apply(v, om, DT); t += DT; robot.check_collision(obstacles, t)
    if rec == 0:
        if abs(v) < 0.05 and dist > GOAL_THRESHOLD:
            stuck += 1
            if stuck >= 15: rec, stuck = 12, 0
        else: stuck = 0
    if dist < GOAL_THRESHOLD:
        if wp < len(WAYPOINTS)-1: wp += 1
        else: break
print(f"run: {len(log)} steps, t={t:.1f}s, final wp {wp+1}, collided={robot.collided}")
traj = np.array([[l[1], l[2]] for l in log])
cand = [i for i, l in enumerate(log) if l[4] is not None]
picks = [cand[int(f*(len(cand)-1))] for f in (0.12, 0.38, 0.62, 0.85)]
fig, axs = plt.subplots(1, 4, figsize=(18, 5.0))
for ax, i in zip(axs, picks):
    tt, x, y, th, X, w_i = log[i]
    for o in obstacles:
        if o.is_dynamic:
            cs = np.array(o.corners(tt + 30*DT)); ax.add_patch(Polygon(cs, fc="none", ec="#e08a00", ls=":", lw=1))
        cs = np.array(o.corners(tt)); ax.add_patch(Polygon(cs, fc="#ffe0b3" if o.is_dynamic else "#f4a6a6", ec="#c46a00", lw=1.2))
    ax.plot(traj[:i+1, 0], traj[:i+1, 1], color="#1f4ea8", lw=2, label="executed path")
    ax.plot(X[0], X[1], "o-", color="#2ca02c", ms=2.5, lw=1.4, label="predicted horizon (3 s)")
    ax.add_patch(Circle((x, y), 0.4, fc="#cfe0fb", ec="#1f4ea8", lw=1.5))
    ax.plot([x, x+0.6*math.cos(th)], [y, y+0.6*math.sin(th)], color="#1f4ea8", lw=2)
    for j, p in enumerate(WAYPOINTS):
        ax.plot(p[0], p[1], "x", color="red" if j == w_i else "#aaa", ms=10, mew=2.5)
    ax.set_xlim(XLIM); ax.set_ylim(YLIM); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(f"t = {tt:.1f} s  (→ waypoint {w_i+1})", fontsize=12); ax.set_xlabel("x [m]")
axs[0].set_ylabel("y [m]"); axs[0].legend(loc="upper left", fontsize=8.5)
fig.suptitle(f"Receding-horizon snapshots, {scene} scene (dotted = obstacle pose at end of horizon)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_horizon_snapshots_{scene}.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("fig3 done")
