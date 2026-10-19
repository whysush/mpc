# MPC Motion Planning — Project Context & Handoff

> **Purpose of this document:** Full context handoff for continuing this project
> on another machine (e.g. with Claude Code). It captures the project spec, the
> current state of the code, every major design decision and dead-end from the
> development conversation, and what's left to do. Read this first before
> touching the code.

---

## 1. What the project is

An end-to-end **Nonlinear Model Predictive Control (MPC)** motion planner for a
nonholonomic (unicycle) mobile robot navigating a cluttered 2D environment.
Built from scratch with **Pygame** (simulation + visualization) and **CasADi**
(symbolic optimization + IPOPT solver). This is a guided 3-week project.

### The robot model (unicycle / differential-drive kinematics)
```
x_{k+1}     = x_k + v_k * cos(theta_k) * dt
y_{k+1}     = y_k + v_k * sin(theta_k) * dt
theta_{k+1} = theta_k + omega_k * dt
```
State `X = [x, y, theta]`, control `U = [v, omega]`.

### Spec deliverables (from the guide's PDF)
- **Week 1:** Pygame 15×15 m arena, oriented-box obstacles, forward-integrating
  unicycle model, keyboard control + collision checking.
- **Week 2:** CasADi `Opti()` OCP. Horizon (example N=20, dt=0.1). Decision vars
  X (3×N+1), U (2×N). Cost `J = Σ‖x_i − x_goal‖²_Q + Σ‖u_i‖²_R`. Hard actuator
  bounds `v_min≤v≤v_max`, `ω_min≤ω≤ω_max`.
- **Week 3:** Per-step algebraic collision-avoidance constraints
  (`((x−x_obs)/a)² + ((y−y_obs)/b)² > 1` or a signed-distance field), CasADi-IPOPT
  solver embedded in the Pygame loop, extract `u_0*` each tick, re-solve next
  tick, draw the predicted horizon. **Deliverable: robot smoothly crosses the
  congested layout to a target without touching any box; modular code +
  performance plots + video.**

### Key spec constraint
The MPC itself must do the navigation — **no global planner (A*, etc.)** was
specified. The MPC tracks the goal pose directly. (We tried adding A*; it was
off-spec and removed — see §5.)

---

## 2. Configuration handed down by the guide

Stored in `config.yaml` (loaded by `config.py`). **These exact values were
specified — do not change them without reason:**

```yaml
robot:
  radius: 0.4            # [m]  -- NOTE: this is a BIG robot
start_pose: [1.0, 1.0, 0.0]   # [x, y, theta]
waypoints:
  - [5.0, 2.0, 1.57]
  - [10.0, 7.0, 1.57]
  - [5.0, 12.0, 3.14]
  - [13.0, 13.0, 0.0]
simulation:
  goal_threshold: 0.5    # [m]
control:
  v_min: 0.0
  v_max: 3.0
  omega_max: 1.0
  a_max: 1.5
  omega_dot_max: 1.5
visualization:
  xlim: [-1, 15]
  ylim: [-1, 15]
  show_predictions: true
  show_obstacles_paramterization: true   # (guide's spelling)
```

Waypoints are `[x, y, theta]` — they include a target heading.

---

## 3. Obstacle data (encrypted .obs files)

Two scenes, loaded via `utils.py` (`load_obstacles`, **DO NOT MODIFY** — it's the
guide's decryption code). The `.obs` files are base64 + xor + zlib + pickle.

- **static.obs** — 13 fixed oriented boxes (`v=0, omega=0`).
- **dynamic.obs** — 20 boxes that **translate** (along their initial heading at
  speed `v`) and **rotate** (rate `omega`). This is the challenging scene.

`Obstacle` namedtuple fields: `(cx, cy, width, height, angle, v, omega)`.

The obstacle set is intentionally adversarial ("challenging cases for your MPC"):
it includes a **4 m diagonal wall at (7,4)** and a **box at (8.5,13) backed
against the arena's top edge** — both create local-minimum traps (see §5/§6).

---

## 4. Current code architecture

All files in the project root. Run with `python run_mpc.py static` or
`python run_mpc.py dynamic`.

| File | Role |
|------|------|
| `config.yaml` | The guide's spec values (source of truth) |
| `config.py` | Loads config.yaml; exposes constants + derived display/scale + colors |
| `geometry.py` | world↔screen coordinate conversion (offset for xlim/ylim = -1) |
| `utils.py` | **Guide's file, do not modify.** Decrypts .obs files |
| `obstacles.py` | `Obstacle` class with **time-varying** pose; signed-distance fn; `load_scene()` |
| `robot.py` | Unicycle model, collision check, drawing |
| `controller.py` | **The MPC** (CasADi Opti + IPOPT) |
| `run_mpc.py` | Main demo: closed loop, scene flag, recovery, HUD, perf plots |
| `main.py` | Week-1 manual keyboard driving (optional) |
| `run_click.py` | Click-to-set-goal interactive demo (optional) |
| `requirements.txt` | pip deps |

### The MPC (`controller.py`) — current design
- Horizon **N=30**, dt=0.1 (≈9 m reach at v_max).
- Decision vars X (3×N+1), U (2×N), plus obstacle slack S (n_obs × N+1).
- Unicycle dynamics as equality constraints.
- Control bounds: v∈[0,3], ω∈[−1,1].
- **Acceleration limits:** per-step Δv bounded by a_max·dt, Δω by omega_dot_max·dt.
- **Obstacle avoidance:** for every obstacle, every step k, signed-distance
  `d(X_k) ≥ (robot_radius + SAFE_MARGIN) − S[i,k]`, with `S ≥ 0`.
  - `SAFE_MARGIN = 0.20` (clearance beyond the 0.4 m radius; also cushions the
    ~0.3 m the robot moves between knots, mitigating tunnelling).
  - For dynamic obstacles, the pose is evaluated at `t = t0 + k·dt` (predicted
    future position) — this is genuine time-varying collision avoidance.
- **Cost:** running position `Q_pos=6`, terminal position `Q_term=60` (dominant,
  prevents the "stay put" minimum), terminal heading `Q_theta=2`, control effort
  `R_v=0.02 / R_omega=0.05`, control-rate smoothing `R_dv=0.3`, and an
  **exact-penalty on obstacle slack: quadratic `W_SLACK=20000` + linear
  `W_SLACK_LIN=8000`**. The linear term is essential — it makes even tiny
  intrusions costly so the solver respects clearance instead of corner-cutting.
- **Solver:** IPOPT, `max_iter=200`, `acceptable_tol=1e-3`, warm-start enabled,
  `mu_strategy=adaptive`.
- **Warm start:** previous solution shifted forward one step (receding-horizon).
- On solver failure: returns the last good control (graceful fallback).

### The closed loop (`run_mpc.py`)
- Loads scene, builds controller, runs the Pygame loop.
- Each tick: solve MPC for current waypoint → apply `u_0` → step robot → advance
  sim clock (drives dynamic obstacles) → check collision → log.
- Advances to next waypoint when within `goal_threshold`.
- **Stuck-recovery behavior (see §6)** — the piece that guarantees completion.
- Draws: grid, obstacles (+ parameterization), waypoint markers (with heading),
  trail, predicted horizon, robot. Saves `mpc_performance_<scene>.png` at the end.

---

## 5. Major design decisions & DEAD ENDS (important — don't repeat these)

The development path had several wrong turns. Recorded so they aren't retried:

1. **A\* global planner — TRIED AND REMOVED.** We added `planner.py` (A* on an
   inflated occupancy grid + pure-pursuit "carrot" tracking) to fix the robot
   freezing. It worked reliably BUT it's **off-spec** — the project requires the
   MPC itself to navigate, not follow a pre-computed path. It reduced the MPC to
   a path-follower. **Deleted.** Do not re-add unless the guide approves.

2. **Longer horizon (N=45, N=50) — MADE IT WORSE.** Intuition said a longer
   horizon would see around walls. Empirically, N=35/45 stalled *earlier* (at
   WP2) and were much slower. Larger N gives the solver more room to find the
   "stay put" local minimum. **N=30 is the sweet spot.**

3. **Midpoint anti-tunnelling constraints — REMOVED.** Added midpoint samples
   between knots to stop the robot clipping thin boxes. Roughly doubled solve
   time. Replaced by a larger `SAFE_MARGIN` (0.20) which cushions the inter-knot
   gap more cheaply. Knot-only constraints = spec-aligned.

4. **Nearest-K obstacle filtering — FIRST ATTEMPT REVERTED, LATER REDONE
   CORRECTLY.** The first try constrained the K=7 closest obstacles with a
   smooth `active*` multiplier — that added nonlinearity, barely helped, and
   made stalls worse. The multiplier was the mistake. The **current** design
   (see §7.1 / `controller.py`) drops it: it hard-constrains the K=10 nearest
   obstacles, chosen against the *predicted horizon path* (not the current
   point) so obstacles ahead stay constrained and the active set is stable
   frame-to-frame. That version is fast AND keeps all-waypoints/0-collisions on
   both scenes. Selecting by current position alone (jittery active set) *does*
   destabilize the trap-prone scenes — path-aware selection is the key.

5. **Over-aggressive weight tuning — caused see-sawing.** Cranking W_SLACK very
   high made the robot freeze (too scared to move); too low made it clip
   corners. The fix was the **exact-penalty (linear + quadratic slack)**, not
   brute-force weights.

### The one genuinely good fix that survived
The **exact-penalty linear slack term** (`W_SLACK_LIN`). A pure quadratic slack
penalty is cheap for small violations, so the solver shaves corners. Adding a
large *linear* term makes any intrusion expensive — the standard trick to make a
soft constraint behave like a hard one.

---

## 6. THE CORE PROBLEM and the final fix

**Diagnosis (took many iterations to pin down):** A pure local MPC traps in
**local minima** on this map. On legs where a box sits on the straight line to
the goal — especially **WP4 (13,13)**, where the box at (8.5,13) is backed
against the arena's top edge so there's no "over the top" route — moving toward
the goal is blocked and any escape briefly *increases* cost, so the optimizer
returns `v≈0` and the robot freezes. This is inherent to local optimization, not
a tuning bug. Verified empirically: every pure-MPC config stalled on ≥1 leg
(WP2, WP3, or WP4 depending on settings).

**The fix: stuck-recovery behavior** (same concept as Nav2's recovery
behaviors). The MPC remains the planner. Logic in `run_mpc.py`:
- Track consecutive low-speed steps (`|v| < 0.05` while not at goal).
- After ~15 stuck steps (~1.5 s), enter a **recovery burst** (12 steps):
  rotate toward the goal bearing; creep forward *only* if the next step stays
  collision-free (so recovery never drives into a box).
- Then hand control back to the MPC — now with a different heading that lets it
  find a descent direction out of the trap.

**Verified result (headless):** Reaches **all 4 waypoints**, WP1→WP2→WP3→WP4,
**0 collisions**, recovery triggered 3 times. This was the user's hard
requirement: all waypoints must be reached.

---

## 7. Known limitations / open items

1. **Solve speed: RESOLVED for the 10 Hz target.** Was ~200 ms/step on the
   dynamic scene (20 obstacles). Two changes in `controller.py` fixed it, both
   license-free with no compile cost:
   - **`expand=True`** on the IPOPT solver — MX→SX graph, ~2× cheaper function
     evaluation each solve, solution unchanged.
   - **K=10 nearest-obstacle constraints, selected along the predicted path**
     — the KKT system scaled with n_obs (×N knots) and dominated the solve;
     constraining only the nearest 10 (parameterized geometry, one compiled
     solver reused as the active set changes) cuts that cost.
   - **Verified headless (both scenes, all 4 waypoints, 0 collisions):**
     static ~17 Hz (≈57 ms), dynamic ~12–13 Hz (≈80 ms). Median/p90 are well
     under the 100 ms (10 Hz) budget. Note a ~240–410 ms spike at waypoint
     *switches* (stale warm start on the first solve of a new leg) — brief and
     expected; doesn't affect the sustained rate.
   - **Further headroom, if ever needed (20 Hz+ on the dynamic scene):** HSL
     `ma57`/`ma27` linear solver (one-line `"linear_solver":"ma57"`, ~2–5×, but
     needs a free academic HSL license), or **fatrop** (CasADi-native,
     structure-exploiting, no license) — fatrop needs the OCP reformulated with
     a strictly stage-ordered variable layout (state-augment to remove the
     cross-stage accel constraint), which is a larger rewrite. JIT codegen was
     tried and rejected: IPOPT's linear algebra (not function eval) is the
     bottleneck, so JIT bought nothing while adding a ~100 s cold-compile.
2. **Dynamic scene NOT yet verified with recovery.** Only `static` was tested
   end-to-end. Run `python run_mpc.py dynamic` and check behavior.
3. **No clean-environment install test** was done — `requirements.txt` versions
   are conservative minimums, not pinned/verified on a fresh venv.
4. **Video log** (Week-3 deliverable) not yet produced.

---

## 8. How to run

```bash
pip install -r requirements.txt        # casadi, numpy, pygame, matplotlib, pyyaml
python run_mpc.py static               # 13 static obstacles
python run_mpc.py dynamic              # 20 moving+rotating obstacles
# main.py  -> manual keyboard driving (Week 1)
# run_click.py -> click to set a goal
```
Controls in the demo: `R` reset, `ESC` quit. Performance plot saved as
`mpc_performance_<scene>.png` when the tour completes.

Files that must travel together: all `.py` + `config.yaml` + `static.obs` +
`dynamic.obs`. (`utils.py` is the guide's decryptor — required, do not edit.)

---

## 9. Suggested next steps (priority order)

1. Run & verify the **dynamic** scene; tune recovery/weights if it stalls.
2. Speed pass: wire in `ma57` (guarded fallback to MUMPS) and/or profile.
3. Record a video log for the Week-3 deliverable.
4. (Optional) Clean-venv install test; pin exact dependency versions.

---

*This document reflects the project state as of the handoff. The user's hard
requirement — reach all 4 waypoints — is met for the static scene. Honesty note
for whoever continues: the development churned through several off-spec
approaches (§5) before landing on MPC + recovery; don't reintroduce them.*
