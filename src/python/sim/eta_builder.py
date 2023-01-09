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
    eta, _ = rw.read_eta_last_file(out_path, xm.shape[0], xm.shape[1])

    return eta


def center_line(xm, ym, config, eta_i):

    eta = tanhmin(np.abs(xm) - config["initRadius"], config["interfaceWidth"])
    return eta.astype(complex) * config["initEta"]
