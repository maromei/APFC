import numpy as np

from . import read_write as rw


def tanhmin(radius: np.array, eps: float) -> np.array:
    return 0.5 * (1.0 + np.tanh(-3.0 * radius / eps))


def single_grain(xm, ym, r, eps, init_eta) -> np.array:

    radius: np.array = np.sqrt(xm**2 + ym**2) - r
    radius = tanhmin(radius, eps)

    return radius * init_eta


def center_line(xm, r, eps, init_eta):

    eta = tanhmin(np.abs(xm) - r, eps)
    return eta * init_eta


def init_eta_height(config, use_pm=False):

    t = config["t"]
    v = config["v"]
    n0 = config["n0"]
    dB0 = config["dB0"]

    if not use_pm:
        config["initEta"] = 4.0 * t / (45.0 * v)
        return

    if n0 > t:
        config["initEta"] = (t - np.sqrt(t**2 - 15.0 * v * dB0)) / 15.0 * v
    else:
        config["initEta"] = (t + np.sqrt(t**2 - 15.0 * v * dB0)) / 15.0 * v
