from config import PX_PER_M, WINDOW_PX, XLIM, YLIM


def world_to_screen(x, y):
    sx = (x - XLIM[0]) * PX_PER_M
    sy = WINDOW_PX - (y - YLIM[0]) * PX_PER_M
    return int(sx), int(sy)


def screen_to_world(sx, sy):
    x = sx / PX_PER_M + XLIM[0]
    y = (WINDOW_PX - sy) / PX_PER_M + YLIM[0]
    return x, y


def scale(meters):
    return int(meters * PX_PER_M)
