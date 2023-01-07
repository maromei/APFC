from typing import Callable

import numpy as np


class FFTSim:

    # Default config for db0 0.044
    A: float = 0.98
    B: float = 0.044
    C: float = -0.5
    D: float = 0.33333

    xlim: int = 400
    pt_count = 1000
    dt: float = 0.5

    G: np.array = None

    etas: np.array = None
    g_sq_hat: np.array = None

    x: np.array = None
    xm: np.array = None
    ym: np.array = None

    kx: np.array = None
    kxm: np.array = None
    kym: np.array = None

    eta_count: int = 0

    def __init__(self, config: dict, eta_builder: Callable):

        #########################
        ## VARIABLE ASSIGNMENT ##
        #########################

        self.A = config.get("A", self.A)
        self.B = config.get("B", self.B)
        self.C = config.get("C", self.C)
        self.D = config.get("D", self.D)

        self.xlim = config.get("xlim", self.xlim)
        self.pt_count = config.get("numPts", self.pt_count)
        self.dt = config.get("dt", self.dt)

        self.G = np.array(config["G"])
        self.eta_count = self.G.shape[0]

        ##############
        ## BUILDING ##
        ##############

        self.build(eta_builder, config)

    ########################
    ## BUILDING FUNCTIONS ##
    ########################

    def build_grid(self):

        self.x = np.linspace(-self.xlim, self.xlim, self.pt_count)
        self.xm, self.ym = np.meshgrid(self.x, self.x)

        self.kx = np.fft.fftfreq(len(self.x), np.diff(self.x)[0])
        self.kxm, self.kym = np.meshgrid(self.kx, self.kx)

    def build_eta(self, eta_builder: Callable, config: dict):
        """
        Needs build_grid() to be called before this function!
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=complex
        )

        for eta_i in range(self.eta_count):
            self.etas[eta_i:, :] += eta_builder(self.xm, self.ym, config)

    def build_gsq_hat(self):
        """
        Needs build_eta() to be called before this function!
        """

        self.g_sq_hat = np.zeros(self.etas.shape, dtype=complex)
        for eta_i in range(self.eta_count):
            self.g_sq_hat[eta_i, :, :] = self.g_sq_hat_fnc(eta_i)

    def build(self, eta_builder: Callable, config: dict):

        self.build_grid()
        self.build_eta(eta_builder, config)
        self.build_gsq_hat()

    ###################
    ## SIM FUNCTIONS ##
    ###################

    def g_sq_hat_fnc(self, eta_i: int) -> np.array:

        ret = self.kxm**2 + self.kym**2
        ret += 2.0 * self.G[eta_i, 0] * self.kxm
        ret += 2.0 * self.G[eta_i, 1] * self.kym

        return ret**2

    def amp_abs_sq_sum(self, eta_i: int) -> np.array:

        sum_ = np.zeros(self.etas[0].shape, dtype=complex)
        for eta_j in range(self.eta_count):

            is_eta_i = int(eta_i == eta_j)
            sum_ += (2.0 - is_eta_i) * self.etas[eta_j] * np.conj(self.etas[eta_j])

        return sum_

    def n_hat(self, eta_i: int) -> np.array:

        poss_eta_is = set([i for i in range(self.eta_count)])
        other_etas = list(poss_eta_is.difference({eta_i}))

        eta_conj1 = np.conj(self.etas[other_etas[0]])
        eta_conj2 = np.conj(self.etas[other_etas[1]])

        n = 3.0 * self.D * self.amp_abs_sq_sum(eta_i) * self.etas[eta_i]
        n += 2.0 * self.C * eta_conj1 * eta_conj2
        n = np.fft.fft2(n)

        return -1.0 * n * np.linalg.norm(self.G[eta_i]) ** 2

    def lagr_hat(self, eta_i: int):

        lagr = self.A * self.g_sq_hat[eta_i] + self.B
        return -1.0 * lagr * np.linalg.norm(self.G[eta_i]) ** 2

    def eta_routine(self, eta_i: int) -> np.array:

        lagr = self.lagr_hat(eta_i)
        n = self.n_hat(eta_i)

        exp_lagr = np.exp(lagr * self.dt)

        n_eta = exp_lagr * np.fft.fft2(self.etas[eta_i])
        n_eta += ((exp_lagr - 1.0) / lagr) * n

        return np.fft.ifft2(n_eta)

    def run_one_step(self):

        n_etas = np.zeros(self.etas.shape, dtype=complex)

        for eta_i in range(self.eta_count):
            n_etas[eta_i, :, :] = self.eta_routine(eta_i)

        self.etas = n_etas

    ##################
    ## IO FUNCTIONS ##
    ##################

    def reset_out_files(self, out_path: str):

        for eta_i in range(self.eta_count):
            with open(f"{out_path}/out_{eta_i}.txt", "w") as f:
                f.write("")

    def write(self, out_path: str):

        for eta_i in range(self.eta_count):

            eta_out_path = f"{out_path}/out_{eta_i}.txt"

            out = self.etas[eta_i].flatten()
            out = out.astype(str)
            out = ",".join(out.tolist())
            out += "\n"

            with open(eta_out_path, "a") as f:
                f.write(out)
