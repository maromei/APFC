import numpy as np
from dataclasses import dataclass
from . import eta_builder

config = {
    "theta": 0.0,
    "numPts": 1000,
    "numPts_y": 1,
    "xlim": 1500,
    "Bx": 0.98,
    "n0": -0.03,
    "v": 0.3333333333333333,
    "t": 0.5,
    "dB0": 0.012,
    "dt": 0.75,
    "initRadius": 50.0,
    "interfaceWidth": 10.0,
    "writeEvery": 200,
    "numT": 30000,
    "G": [[-0.8660254037844386, -0.5], [0.0, 1.0], [0.8660254037844386, -0.5]],
}

###
###


def g_sq_hat_fnc(kxm, kym, G, eta_i):

    ret = kxm**2 + kym**2
    ret += 2.0 * G[eta_i, 0] * kxm
    ret += 2.0 * G[eta_i, 1] * kym

    return ret**2


def sim(config, theta, use_grain=False):

    config = config.copy()

    ################
    ## BUILD GRID ##
    ################

    x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
    y = np.linspace(-config["xlim"], config["xlim"], config["numPts_y"])

    numPts = config["numPts"]
    numPts_y = config.get("numPts_y", numPts)

    if numPts <= 1:
        x = np.array([0.0])

    if numPts_y <= 1:
        y = np.array([0.0])

    if numPts <= 1:
        kx = np.array([0.0])
    else:
        kx = np.fft.fftfreq(len(x), np.abs(np.diff(x)[0]))

    if numPts_y <= 1:
        ky = np.array([0.0])
    else:
        ky = np.fft.fftfreq(len(y), np.abs(np.diff(x)[0]))

    xm, ym = np.meshgrid(x, y)
    kxm, kym = np.meshgrid(kx, ky)

    ####################
    ## INIT VARIABLES ##
    ####################

    t = config["t"]
    v = config["v"]

    # fmt: off
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # fmt: on

    G = np.array(config["G"])
    for eta_i in range(eta_count):
        G[eta_i] = rot.dot(G[eta_i])
    eta_count = G.shape[0]

    init_eta = 4.0 * t / (45.0 * v)

    #####################
    ## BUILD VARIABLES ##
    #####################

    etas = np.zeros((eta_count, xm.shape[0], xm.shape[1]), dtype=float)

    for eta_i in range(eta_count):
        if use_grain:
            etas[eta_i] = eta_builder.single_grain(
                xm, ym, config["initRadius"], config["interfaceWidth"], init_eta
            )
        else:
            etas[eta_i] = eta_builder.center_line(
                xm, config["initRadius"], config["interfaceWidth"], init_eta
            )

    n0 = np.zeros(xm.shape, dtype=float)

    g_sq_hat = np.zeros(etas.shape, dtype=float)
    for eta_i in range(eta_count):
        g_sq_hat[eta_i] = g_sq_hat_fnc(kxm, kym, G, eta_i)

    laplace_op = -(kxm**2) + kym**2

    ############
    ## n0 SIM ##
    ############
