from typing import Callable

import numpy as np
from scipy.sparse.linalg import cg


class FFTSim:

    # Default config for db0 0.044
    A: float = 0.98
    D: float = 0.33333
    lbd: float = 1.024
    t: float = 0.5
    v: float = 1.0 / 3.0
    dB0: float = 0.044

    a: float = 1.0

    init_n0: float = 0.0

    xlim: int = 400
    pt_count_x = 1000
    pt_count_y = 1000
    dt: float = 0.5

    G: np.array = None

    etas: np.array = None
    n0: np.array = None
    g_sq_hat: np.array = None
    laplace_op: np.array = None

    xm: np.array = None
    ym: np.array = None

    kxm: np.array = None
    kym: np.array = None

    eta_count: int = 0

    def __init__(self, config: dict, eta_builder: Callable):

        #########################
        ## VARIABLE ASSIGNMENT ##
        #########################

        self.A = config.get("A", self.A)
        self.D = config.get("D", self.D)
        self.t = config.get("t", self.t)
        self.v = config.get("v", self.v)
        self.init_n0 = config.get("n0", self.init_n0)

        self.dB0 = config.get("dB0", self.dB0)
        self.lbd = self.A + self.dB0

        self.xlim = config.get("xlim", self.xlim)
        self.ylim = config.get("ylim")
        self.pt_count_x = config.get("numPts", self.pt_count_x)
        self.pt_count_y = config.get("numPts_y", self.pt_count_x)
        self.dt = config.get("dt", self.dt)

        self.G = np.array(config["G"])
        self.eta_count = self.G.shape[0]

        self.config = config

        ##############
        ## BUILDING ##
        ##############

        self.build(eta_builder, config)

    ########################
    ## BUILDING FUNCTIONS ##
    ########################

    def build_grid(self):

        x = np.linspace(-self.xlim, self.xlim, self.pt_count_x)
        y = np.linspace(-self.xlim, self.xlim, self.pt_count_y)
        self.xm, self.ym = np.meshgrid(x, y)

        dx = np.diff(x)[0]

        freq_x = np.fft.fftfreq(len(x), dx)

        if self.pt_count_y <= 1:
            freq_y = [0.0]
        else:
            dy = np.diff(y)[0]
            freq_y = np.fft.fftfreq(len(y), dy)

        self.kxm, self.kym = np.meshgrid(freq_x, freq_y)

    def build_eta(self, eta_builder: Callable, config: dict):
        """
        Needs build_grid() to be called before this function!
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=float
        )

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += eta_builder(self.xm, self.ym, config, eta_i)

    def build_gsq_hat(self):
        """
        Needs build_eta() to be called before this function!
        """

        shape = (self.eta_count, self.kxm.shape[0], self.kxm.shape[1])

        self.g_sq_hat = np.zeros(shape, dtype=float)
        for eta_i in range(self.eta_count):
            self.g_sq_hat[eta_i, :, :] = self.g_sq_hat_fnc(eta_i)

    def build_n0(self):

        self.n0 = np.ones(self.xm.shape, dtype=float) * self.init_n0
        self.n0_old = np.ones(self.xm.shape, dtype=float) * self.init_n0

    def build_laplace_op(self):
        """
        Needs build_grid() to be called before this function!
        """

        self.laplace_op = -(self.kxm**2 + self.kym**2)

    def build(self, eta_builder: Callable, config: dict):

        self.build_grid()
        self.build_eta(eta_builder, config)
        self.build_gsq_hat()
        self.build_laplace_op()
        self.build_n0()

    #########################
    ## SIM FUNCTIONS - ETA ##
    #########################

    def g_sq_hat_fnc(self, eta_i: int) -> np.array:

        ret = self.kxm**2 + self.kym**2
        ret += 2.0 * self.G[eta_i, 0] * self.kxm
        ret += 2.0 * self.G[eta_i, 1] * self.kym

        return ret**2

    def amp_abs_sq_sum(self, eta_i: int) -> np.array:

        sum_ = np.zeros(self.etas[0].shape, dtype=float)
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
        n += 2.0 * self.C(self.n0) * eta_conj1 * eta_conj2
        n = np.fft.fft2(n)
        return -1.0 * n * np.linalg.norm(self.G[eta_i]) ** 2

    def lagr_hat(self, eta_i: int):

        lagr = self.A * self.g_sq_hat[eta_i] + self.B(self.n0)
        return -1.0 * lagr * np.linalg.norm(self.G[eta_i]) ** 2

    def eta_routine(self, eta_i: int) -> np.array:

        lagr = self.lagr_hat(eta_i)
        n = self.n_hat(eta_i)

        exp_lagr = np.exp(lagr * self.dt)

        n_eta = exp_lagr * np.fft.fft2(self.etas[eta_i])
        n_eta += ((exp_lagr - 1.0) / lagr) * n

        return np.real(np.fft.ifft2(n_eta, s=self.etas[0].shape))

    ########################
    ## SIM FUNCTIONS - N0 ##
    ########################

    def B(self, n0):

        ret = self.dB0
        ret -= 2.0 * self.t * self.n0
        ret += 3.0 * self.v * self.n0**2

        return ret

    def C(self, n0):

        return -self.t + 3.0 * self.v * n0

    def get_eta_prod(self):

        eta_prod = np.ones(self.etas[0].shape, dtype=complex)

        for eta_i in range(self.eta_count):
            eta_prod *= self.etas[eta_i]

        eta_prod += np.conj(eta_prod)

        return 2.0 * eta_prod

    def get_phi(self):

        eta_sum = np.zeros(self.etas[0].shape, dtype=complex)
        for eta_i in range(self.eta_count):
            eta_sum += self.etas[eta_i] * np.conj(self.etas[eta_i])

        return 2.0 * eta_sum

    def n0_routine_solve(self):

        phi = self.get_phi()
        eta_prod = self.get_eta_prod()

        lagr = phi * 3.0 * self.v + self.dB0

        n = -phi * self.t
        n += 3.0 * self.v * eta_prod
        n -= self.t * self.n0**2
        n += self.v * self.n0**3

        n = np.fft.fft2(n)

        n_n0 = np.zeros(n.shape, dtype=complex)

        for i in range(n.shape[0]):
            for j in range(n.shape[1]):

                A = np.array(
                    [[1.0 / self.dt, -self.laplace_op[i, j]], [-lagr[i, j], 1]]
                )

                b = np.array([self.n0[i, j] / self.dt, n[i, j]])

                # out, _ = cg(A, b)
                out = np.linalg.solve(A, b)

                n_n0[i, j] = out[0]

        return np.real(np.fft.ifft2(n_n0, s=self.etas[0].shape))

    def n0_routine(self):

        phi = self.get_phi()
        eta_prod = self.get_eta_prod()

        lagr = phi * 3.0 * self.v + self.dB0

        n = -phi * self.t
        n += 3.0 * self.v * eta_prod
        n -= self.t * self.n0**2
        n += self.v * self.n0**3

        n = np.fft.fft2(n)

        denom = 1.0 - self.dt * self.laplace_op * lagr
        n_n0 = np.fft.fft2(self.n0) + self.dt * self.laplace_op * n
        n_n0 = n_n0 / denom

        return np.real(np.fft.ifft2(n_n0, s=self.etas[0].shape))

    ###################
    ## SIM FUNCTIONS ##
    ###################

    def run_one_step(self):

        self.n0_old = self.n0.copy()
        self.n0 = self.n0_routine()

        if self.config["keepEtaConst"]:
            return

        n_etas = np.zeros(self.etas.shape, dtype=float)

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

        with open(f"{out_path}/n0.txt", "w") as f:
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

        with open(f"{out_path}/n0.txt", "a") as f:

            out = self.n0.flatten()
            out = out.astype(str)
            out = ",".join(out.tolist())
            out += "\n"

            f.write(out)
