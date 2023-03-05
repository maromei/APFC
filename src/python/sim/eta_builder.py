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


def load_n0_from_file(xm, config):
    out_path = f"{config['sim_path']}/n0.txt"
    n0, _ = rw.read_eta_last_file(out_path, xm.shape[0], xm.shape[1], dtype=float)
    return n0


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


def init_eta_height(config, use_pm=False, use_n0=False):

    t = config["t"]
    v = config["v"]
    n0 = config["n0"] if use_n0 else 0.0
    dB0 = config["dB0"]

    if not use_pm:
        config["initEta"] = 4.0 * t / (45.0 * v)
        return

    B = dB0 - 2 * t * n0 + 3 * v * n0**2
    C = -t + 3 * v * n0

    sign = -1 if n0 > t else 1

    config["initEta"] = (-C + sign * np.sqrt(C**2 - 15 * v * B)) / (15 * v)


def init_n0_height(
    config: dict,
    x0: float = 0.0,
    ATOL: float = 1e-5,
    RTOL: float = 1e-5,
    MAXITER: int = 1000000,
):
    """
    uses initEta in config to set equilibrium n0

    uses newton method
    """

    phi = 6 * config["initEta"] ** 2
    p = 4 * config["initEta"] ** 3

    a = config["dB0"] + config["Bx"] + 3 * config["v"] * phi
    c = phi * config["t"]
    d = 3 * config["v"] * p

    def f(n0, a, v, t, c, d):
        return a * n0 - t * n0**2 + v * n0**3 + d - c

    def df(n0, a, v, t):
        return a - 2 * t * n0 + 3 * v * n0**2

    xold = x0 + 100.0
    xnew = x0

    k = 0
    while np.abs(xnew - xold) > np.abs(xnew) * RTOL + ATOL:

        xold = xnew
        fval = f(xold, a, config["v"], config["t"], c, d)
        dfval = df(xold, a, config["v"], config["t"])

        xnew = xold - fval / dfval
        k += 1

        if k > MAXITER:
            print("REACHED MAX ITER ON n0 CALCULATION!")
            break

    config["n0"] = -0.5 * xnew
