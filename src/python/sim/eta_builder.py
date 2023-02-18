import numpy as np

from . import read_write as rw


def tanhmin(radius: np.array, eps: float) -> np.array:
    return 0.5 * (1.0 + np.tanh(-3.0 * radius / eps))


def single_grain(xm: np.array, ym: np.array, config: dict, eta_i: int) -> np.array:

    radius: np.array = np.sqrt(xm**2 + ym**2) - config["initRadius"]
    radius = tanhmin(radius, config["interfaceWidth"])

    return radius * config["initEta"]


def load_from_file(xm, ym, config, eta_i):

    out_path = f"{config['sim_path']}/out_{eta_i}.txt"
    eta, _ = rw.read_eta_last_file(out_path, xm.shape[0], xm.shape[1], dtype=float)

    return eta


def center_line(xm, ym, config, eta_i):

    eta = tanhmin(np.abs(xm) - config["initRadius"], config["interfaceWidth"])
    return eta * config["initEta"]


def left_line(xm, ym, config, eta_i):

    eta = tanhmin(xm + config["initRadius"], config["interfaceWidth"])
    return eta * config["initEta"]


def init_config(config):

    n0 = config["n0"]
    t = config["t"]
    v = config["v"]
    Bx = config["Bx"]
    dB0 = config["dB0"]

    config["A"] = Bx
    config["B"] = dB0 - 2.0 * t * n0 + 3.0 * v * n0**2
    config["C"] = -t + 3.0 * v * n0
    config["D"] = v


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
