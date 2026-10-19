"""
Configuration loader.

Reads config.yaml (the spec handed down by the guide) and exposes the values
as module-level constants so the rest of the code can keep doing
`from config import V_MAX, ...`. Derived display constants (pixels-per-meter,
window size, colors) live here too.
"""

import os
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG_PATH = os.path.join(_HERE, "config.yaml")

with open(_CFG_PATH, "r") as _f:
    CFG = yaml.safe_load(_f)

# --- robot ---------------------------------------------------------------
ROBOT_RADIUS_M = float(CFG["robot"]["radius"])

# --- poses / waypoints ---------------------------------------------------
# start_pose is [x, y, theta]; waypoints are [x, y, theta].
START_POSE = tuple(CFG["start_pose"])
WAYPOINTS = [tuple(wp) for wp in CFG["waypoints"]]
GOAL_M = (WAYPOINTS[-1][0], WAYPOINTS[-1][1])  # final waypoint (x, y)

# --- simulation ----------------------------------------------------------
GOAL_THRESHOLD = float(CFG["simulation"]["goal_threshold"])
DT = 0.1
FPS = 60

# --- control limits ------------------------------------------------------
V_MIN = float(CFG["control"]["v_min"])
V_MAX = float(CFG["control"]["v_max"])
OMEGA_MAX = float(CFG["control"]["omega_max"])
OMEGA_MIN = -OMEGA_MAX
A_MAX = float(CFG["control"]["a_max"])
OMEGA_DOT_MAX = float(CFG["control"]["omega_dot_max"])

# Manual-driving step sizes (used only by main.py).
V_STEP = 0.1
OMEGA_STEP = 0.1

# --- visualization -------------------------------------------------------
XLIM = tuple(CFG["visualization"]["xlim"])
YLIM = tuple(CFG["visualization"]["ylim"])
SHOW_PREDICTIONS = bool(CFG["visualization"]["show_predictions"])
SHOW_OBSTACLE_PARAMETERIZATION = bool(
    CFG["visualization"]["show_obstacles_paramterization"]
)

# Drawing scale derived from the requested view window so the whole arena
# (plus the negative margins in xlim/ylim) fits on screen.
WINDOW_PX = 800
_view_w = XLIM[1] - XLIM[0]
_view_h = YLIM[1] - YLIM[0]
PX_PER_M = WINDOW_PX / max(_view_w, _view_h)
ARENA_M = max(_view_w, _view_h)

# --- colors --------------------------------------------------------------
C_BG          = (245, 245, 245)
C_GRID        = (220, 220, 220)
C_OBSTACLE    = (240, 128, 128)
C_OBS_EDGE    = (200,  80,  80)
C_DYN_OBS     = (250, 170,  90)
C_DYN_OBS_EDGE = (210, 120,  40)
C_ROBOT       = ( 50, 120, 220)
C_ROBOT_HIT   = (220,  50,  50)
C_HEADING     = ( 20,  20,  20)
C_GOAL        = (220,  30,  30)
C_TRAIL       = ( 80, 140, 230)
C_PRED        = (  0, 170,   0)
C_PARAM       = (120,  60, 160)
C_TEXT        = ( 30,  30,  30)
