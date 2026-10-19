import casadi
import numpy as np

from config import (DT, ROBOT_RADIUS_M, V_MIN, V_MAX, OMEGA_MAX, OMEGA_MIN,
                    A_MAX, OMEGA_DOT_MAX)

N = 30
Q_pos = 6.0
Q_term = 60.0
Q_theta = 2.0
R_v = 0.02
R_omega = 0.05
R_dv = 0.3
W_SLACK = 20000.0
W_SLACK_LIN = 8000.0

SAFE_MARGIN = 0.20
K_NEAR = 10
_FAR_SLOT = [1e4, 1e4, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0]


def _obs_params(o):
    return [o.cx0, o.cy0, o.w / 2.0, o.h / 2.0, o.angle0,
            o._dx, o._dy, o.omega]


def _sd_param(px, py, p, t):
    cx = p[0] + p[5] * t
    cy = p[1] + p[6] * t
    ang = p[4] + p[7] * t
    dx, dy = px - cx, py - cy
    c, s = casadi.cos(-ang), casadi.sin(-ang)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    ax = casadi.fabs(lx) - p[2]
    ay = casadi.fabs(ly) - p[3]
    outside = casadi.sqrt(casadi.fmax(ax, 0) ** 2 + casadi.fmax(ay, 0) ** 2 + 1e-6)
    inside = casadi.fmin(casadi.fmax(ax, ay), 0.0)
    return outside + inside


class MPCController:
    def __init__(self, goal, obstacles):
        self.obstacles = obstacles

        opti = casadi.Opti()
        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)
        P_start = opti.parameter(3)
        P_goal = opti.parameter(3)
        P_t0 = opti.parameter(1)

        K = min(K_NEAR, len(obstacles))
        self.K = K
        P_obs = opti.parameter(8, K)
        S = opti.variable(K, N + 1)

        for k in range(N):
            x, y, th = X[0, k], X[1, k], X[2, k]
            v, om = U[0, k], U[1, k]
            opti.subject_to(X[0, k + 1] == x + v * casadi.cos(th) * DT)
            opti.subject_to(X[1, k + 1] == y + v * casadi.sin(th) * DT)
            opti.subject_to(X[2, k + 1] == th + om * DT)

        opti.subject_to(X[:, 0] == P_start)
        opti.subject_to(opti.bounded(V_MIN, U[0, :], V_MAX))
        opti.subject_to(opti.bounded(OMEGA_MIN, U[1, :], OMEGA_MAX))

        for k in range(N - 1):
            dv = U[0, k + 1] - U[0, k]
            dom = U[1, k + 1] - U[1, k]
            opti.subject_to(opti.bounded(-A_MAX * DT, dv, A_MAX * DT))
            opti.subject_to(opti.bounded(-OMEGA_DOT_MAX * DT, dom,
                                         OMEGA_DOT_MAX * DT))

        safe = ROBOT_RADIUS_M + SAFE_MARGIN
        for i in range(K):
            for k in range(N + 1):
                t = P_t0 + k * DT
                d = _sd_param(X[0, k], X[1, k], P_obs[:, i], t)
                opti.subject_to(d >= safe - S[i, k])
                opti.subject_to(S[i, k] >= 0)

        cost = 0
        for k in range(N + 1):
            dx = X[0, k] - P_goal[0]
            dy = X[1, k] - P_goal[1]
            cost += Q_pos * (dx * dx + dy * dy)

        dxN = X[0, N] - P_goal[0]
        dyN = X[1, N] - P_goal[1]
        cost += Q_term * (dxN * dxN + dyN * dyN)
        cost += Q_theta * (1 - casadi.cos(X[2, N] - P_goal[2]))

        for k in range(N):
            cost += R_v * U[0, k] ** 2 + R_omega * U[1, k] ** 2
        for k in range(N - 1):
            dv = U[0, k + 1] - U[0, k]
            dw = U[1, k + 1] - U[1, k]
            cost += R_dv * (dv * dv + dw * dw)
        for i in range(K):
            for k in range(N + 1):
                cost += W_SLACK * S[i, k] ** 2 + W_SLACK_LIN * S[i, k]
        opti.minimize(cost)

        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "sb": "yes",
                     "max_iter": 200,
                     "acceptable_tol": 1e-3, "acceptable_iter": 5,
                     "warm_start_init_point": "yes",
                     "mu_strategy": "adaptive"})

        self.opti = opti
        self.X = X
        self.U = U
        self.S = S
        self.P_start = P_start
        self.P_goal = P_goal
        self.P_t0 = P_t0
        self.P_obs = P_obs
        self._obs_p = [_obs_params(o) for o in obstacles]
        self.opti.set_value(self.P_goal, self._goal3(goal))
        self.opti.set_value(self.P_t0, 0.0)
        self._select_obstacles(np.array(self._goal3(goal))[:2], 0.0)
        self.last_X = None
        self.last_U = None

    def _select_obstacles(self, pos, t0, path=None):
        n = len(self.obstacles)
        if n == 0:
            self.opti.set_value(self.P_obs, np.array([_FAR_SLOT] * self.K).T)
            return
        if path is not None:
            pts = [(path[0, k], path[1, k], t0 + k * DT)
                   for k in range(path.shape[1])]
        else:
            pts = [(pos[0], pos[1], t0)]

        def obs_key(i):
            o = self.obstacles[i]
            best = float("inf")
            for px, py, t in pts:
                cx = o.cx0 + o._dx * t
                cy = o.cy0 + o._dy * t
                best = min(best, (cx - px) ** 2 + (cy - py) ** 2)
            return best

        order = sorted(range(n), key=obs_key)
        cols = [self._obs_p[i] for i in order[:self.K]]
        while len(cols) < self.K:
            cols.append(_FAR_SLOT)
        self.opti.set_value(self.P_obs, np.array(cols).T)

    @staticmethod
    def _goal3(goal):
        if goal is None:
            return [0.0, 0.0, 0.0]
        if len(goal) == 2:
            return [goal[0], goal[1], 0.0]
        return [goal[0], goal[1], goal[2]]

    def solve(self, current_state, goal=None, t0=0.0):
        self.opti.set_value(self.P_start, current_state)
        self.opti.set_value(self.P_t0, t0)
        if goal is not None:
            self.opti.set_value(self.P_goal, self._goal3(goal))
        self._select_obstacles(np.asarray(current_state)[:2], t0,
                               path=self.last_X)
        if self.last_X is not None:
            Xw = np.hstack([self.last_X[:, 1:], self.last_X[:, -1:]])
            Uw = np.hstack([self.last_U[:, 1:], self.last_U[:, -1:]])
            self.opti.set_initial(self.X, Xw)
            self.opti.set_initial(self.U, Uw)
        try:
            sol = self.opti.solve()
            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            self.last_X = X_sol
            self.last_U = U_sol
            return U_sol[0, 0], U_sol[1, 0], X_sol
        except RuntimeError:
            if self.last_U is not None:
                return self.last_U[0, 0], self.last_U[1, 0], self.last_X
            return 0.0, 0.0, None
