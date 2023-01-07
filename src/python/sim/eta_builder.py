import numpy as np


def tanhmin(radius: np.array, eps: float) -> np.array:
    return 0.5 * (1.0 + np.tanh(-3.0 * radius / eps))


def single_grain(xm: np.array, ym: np.array, config: dict, eta_i: int) -> np.array:

    radius: np.array = np.sqrt(xm**2 + ym**2) - config["initRadius"]
    radius = tanhmin(radius, config["interfaceWidth"])

    return radius * config["initEta"]


def load_from_file(xm, ym, config, eta_i):
    pass
