import numpy as np
from calculations import initialize, params


class FFTN0Sim:
    """
    This class is supposed to run a FFT simulations with
    :math:`n_0`, where

    For more information see the :ref:`ch:scheme_ampl_n0` section.
    """

    # Default config for db0 0.044
    A: float = 0.98  #: param in eq. :eq:`eqn:apfc_flow_constants`
    D: float = 0.33333  #: param in eq. :eq:`eqn:apfc_flow_constants`
    lbd: float = 1.024  #: :math:`\lambda` param in eq. :eq:`eqn:apfc_flow_constants`
    t: float = 0.5  #: param in eq. :eq:`eqn:apfc_flow_constants`
    v: float = 1.0 / 3.0  #: param in eq. :eq:`eqn:apfc_flow_constants`
    dB0: float = 0.044  #: :math:`\Delta B^0` param in eq. :eq:`eqn:apfc_flow_constants`
    Bx: float = 0.988  #: :math:`B^x` param in eq. :eq:`eqn:apfc_flow_constants`
    beta_sqrt: float = 1  #: :math:`\sqrt\beta` in eq. :eq:`eqn:apfc_flow_constants`

    init_n0: float = 0.0

    #: The bounds of the domain :math:`[-\text{xlim}, \text{xlim}]^2`
    xlim: int = 400
    pt_count_x = 1000  #: number of points in x-direction
    pt_count_y = 1000  #: number of points in y-direction
    dt: float = 0.5  #: timestep

    #: reciprical vector; see eq. :eq:`eqn:apfc_flow_constants`.
    #: Should have shape :code:`(eta_count, 2)`
    G: np.array = None

    #: the amplitudes. Should have
    #: shape :code:`(eta_count, pt_count_x, pt_count_y)`
    etas: np.array = None

    #: average densities. Should have
    #: shape :code:`(pt_count_x, pt_count_y)`
    n0: np.array = None

    #: The :math:`\widehat{\mathcal{G}_m^2}` operator.
    #: See eq. :eq:`eqn:g_sq_op_fourier`
    g_sq_hat: np.array = None

    #: The fourier transformed laplace operator.
    laplace_op: np.array = None

    xm: np.array = None  #: x meshgrid
    ym: np.array = None  #: y meshgrid

    kxm: np.array = None  #: x frequency meshgrid
    kym: np.array = None  #: y frequency meshgrid

    eta_count: int = 0  #: number of etas

    def __init__(self, config: dict, con_sim: bool = False):
        """
        Initializes the simulation. Gets all the parameters from
        the config and initializes all grids.

        Args:
            config (dict): config object
            con_sim (bool): whether the simulations should continue.
                If :code:`True`, it will read the last values
                from a file.
        """

        #########################
        ## VARIABLE ASSIGNMENT ##
        #########################

        self.A = params.A(config)
        self.D = params.D(config)
        self.t = config.get("t", self.t)
        self.v = config.get("v", self.v)
        self.beta_sqrt = np.sqrt(config.get("beta", 1.0))
        self.init_n0 = config.get("n0", self.init_n0)

        self.dB0 = config.get("dB0", self.dB0)
        self.Bx = config.get("Bx")
        self.lbd = self.Bx + self.dB0

        self.xlim = config.get("xlim", self.xlim)
        self.pt_count_x = config.get("numPtsX", self.pt_count_x)
        self.pt_count_y = config.get("numPtsY", self.pt_count_y)
        self.dt = config.get("dt", self.dt)

        self.G = np.array(config["G"])
        self.eta_count = self.G.shape[0]

        self.config = config

        ##############
        ## BUILDING ##
        ##############

        self.build(con_sim, config)

    ########################
    ## BUILDING FUNCTIONS ##
    ########################

    def build_grid(self):
        """
        Initializess the grid.
        """

        x = np.linspace(-self.xlim, self.xlim, self.pt_count_x)

        if self.pt_count_y <= 1:
            y = [0.0]
        else:
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

    def build_eta(self, config: dict):
        """
        Initializes the amplitudes

        Needs build_grid() to be called before this function!

        Args:
            config (dict): config object
        """

        if self.pt_count_y <= 1:
            self.init_eta_center_line(config)
        else:
            self.init_eta_grain(config)

    def build_gsq_hat(self):
        """
        Initializess the :math:`\widehat{\mathcal{G}^2_m}` operator.

        Needs build_eta() to be called before this function!
        """

        shape = (self.eta_count, self.kxm.shape[0], self.kxm.shape[1])

        self.g_sq_hat = np.zeros(shape, dtype=float)
        for eta_i in range(self.eta_count):
            self.g_sq_hat[eta_i, :, :] = self.g_sq_hat_fnc(eta_i)

    def build_n0(self, con_sim: bool, config: dict):
        """
        Initializes :math:`n_0`

        Args:
            con_sim (bool): whether the last densities should be read from a
                file
            config (dict): config object
        """

        if con_sim:
            self.n0 = initialize.load_n0_from_file(self.xm.shape, config)
        else:
            self.n0 = np.ones(self.xm.shape, dtype=float) * self.init_n0

        self.n0_old = self.n0.copy()

    def build_laplace_op(self):
        """
        Builds the laplace operator
        :math:`-(k_x^2 + k_y^2)`

        Needs build_grid() to be called before this function!
        """

        self.laplace_op = -(self.kxm**2 + self.kym**2)

    def build(self, con_sim: bool, config: dict):
        """
        Builds the grids, amplitudes, densities and operators.

        Args:
            con_sim (bool): Whether the simulation should continue
                from the last values.
            config (dict): config object.
        """

        self.build_grid()
        if con_sim:
            self.init_eta_file(config)
        else:
            self.build_eta(config)

        self.build_gsq_hat()
        self.build_laplace_op()
        self.build_n0(con_sim, config)

    ########################
    ## INIT ETA FUNCTIONS ##
    ########################

    def init_eta_grain(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.single_grain` function.

        Args:
            config (dict): config object
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=float
        )

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.single_grain(
                self.xm,
                self.ym,
                config,
            )

    def init_eta_center_line(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.center_line` function.

        Args:
            config (dict): config object
        """

        self.etas = np.zeros(
            (self.eta_count, self.xm.shape[0], self.xm.shape[1]), dtype=float
        )

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.center_line(
                self.xm,
                config,
            )

    def init_eta_file(self, config: dict):
        """
        Initializes the amplitudes with the
        :py:meth:`calculations.initialize.load_eta_from_file` function.

        Args:
            config (dict): config object
        """

        shape = (self.eta_count, self.xm.shape[0], self.xm.shape[1])
        self.etas = np.zeros(shape, dtype=float)

        for eta_i in range(self.eta_count):
            self.etas[eta_i, :, :] += initialize.load_eta_from_file(
                self.xm.shape, config, eta_i
            )

    #########################
    ## SIM FUNCTIONS - ETA ##
    #########################

    def g_sq_hat_fnc(self, eta_i: int) -> np.array:
        """
        Calculated the :math:`\widehat{\mathcal{G}_m^2}`
        For one single amplitude.
        See eq. :eq:`eqn:g_sq_op_fourier`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array: operator
        """

        ret = self.beta_sqrt * (self.kxm**2 + self.kym**2)
        ret += 2.0 * self.G[eta_i, 0] * self.kxm
        ret += 2.0 * self.G[eta_i, 1] * self.kym

        return ret**2

    def amp_abs_sq_sum(self, eta_i: int) -> np.array:
        """
        Computes

        .. math::

            \sum\limits_j | \eta_j |^2 - | \eta_m |^2

        for one amplitude :math:`\eta_m`

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        sum_ = np.zeros(self.etas[0].shape, dtype=float)
        for eta_j in range(self.eta_count):

            is_eta_i = int(eta_i == eta_j)
            sum_ += (2.0 - is_eta_i) * self.etas[eta_j] * np.conj(self.etas[eta_j])

        return sum_

    def n_hat(self, eta_i: int) -> np.array:
        """
        Computes :math:`\widehat{N}`
        for one amplitude.

        See :ref:`ch:scheme_ampl_n0`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        poss_eta_is = set([i for i in range(self.eta_count)])
        other_etas = list(poss_eta_is.difference({eta_i}))

        eta_conj1 = np.conj(self.etas[other_etas[0]])
        eta_conj2 = np.conj(self.etas[other_etas[1]])

        n = 3.0 * self.D * self.amp_abs_sq_sum(eta_i)
        n += 3.0 * self.v * self.n0_old**2
        n -= 2.0 * self.t * self.n0_old
        n *= self.etas[eta_i]

        n += 2.0 * self.C(self.n0_old) * eta_conj1 * eta_conj2

        n = np.fft.fft2(n)

        return -1.0 * n * np.linalg.norm(self.G[eta_i]) ** 2

    def lagr_hat(self, eta_i: int):
        """
        Computes the linear part :math:`\widehat{\mathcal{L}}`
        for one amplitude.

        See :ref:`ch:scheme_ampl_n0`.

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        lagr = self.A * self.g_sq_hat[eta_i] + self.dB0
        return -1.0 * lagr * np.linalg.norm(self.G[eta_i]) ** 2

    def eta_routine(self, eta_i: int) -> np.array:
        """
        Runs one time step for one single amplitude.
        See :ref:`ch:scheme_ampl_n0`

        Args:
            eta_i (int): amplitude index

        Returns:
            np.array:
        """

        lagr = self.lagr_hat(eta_i)
        n = self.n_hat(eta_i)

        denom = 1.0 - self.dt * lagr
        n_eta = np.fft.fft2(self.etas[eta_i]) + self.dt * n
        n_eta = n_eta / denom

        return np.real(np.fft.ifft2(n_eta, s=self.etas[0].shape))

    ########################
    ## SIM FUNCTIONS - N0 ##
    ########################

    def B(self, n0: np.array) -> np.array:
        """
        Calculates the :math:`B` parameter in :eq:`eqn:apfc_flow_constants`.

        Args:
            n0 (np.array):

        Returns:
            np.array:
        """

        ret = self.dB0
        ret -= 2.0 * self.t * n0
        ret += 3.0 * self.v * n0**2

        return ret

    def C(self, n0: np.array) -> np.array:
        """
        Calculates the :math:`C` parameter in :eq:`eqn:apfc_flow_constants`.

        Args:
            n0 (np.array):

        Returns:
            np.array:
        """

        return -self.t + 3.0 * self.v * n0

    def get_eta_prod(self) -> np.array:
        """
        Calculates

        .. math::

            2 \\left( \\prod_m \\eta_m + \\prod_m \\eta_m^* \\right)

        Returns:
            np.array:
        """

        eta_prod = np.ones(self.etas[0].shape, dtype=complex)

        for eta_i in range(self.eta_count):
            eta_prod *= self.etas[eta_i]

        eta_prod += np.conj(eta_prod)

        return 2.0 * eta_prod

    def get_phi(self) -> np.array:
        """
        Calculates :math:`\Phi` in :eq:`eqn:apfc_flow_constants`

        Returns:
            np.array:
        """

        eta_sum = np.zeros(self.etas[0].shape, dtype=complex)
        for eta_i in range(self.eta_count):
            eta_sum += self.etas[eta_i] * np.conj(self.etas[eta_i])

        return 2.0 * eta_sum

    def n0_routine(self) -> np.array:
        """
        Computes the new :math:`n_0`.

        Returns:
            np.array:
        """

        phi = self.get_phi()
        eta_prod = self.get_eta_prod()

        lagr = self.lbd

        n = -phi * self.t
        n += 3.0 * self.v * eta_prod
        n -= self.t * self.n0**2
        n += self.v * self.n0**3
        n += phi * 3.0 * self.v * self.n0

        n = np.fft.fft2(n)

        denom = 1.0 - self.dt * self.laplace_op * lagr
        n_n0 = np.fft.fft2(self.n0) + self.dt * self.laplace_op * n
        n_n0 = n_n0 / denom

        return np.real(np.fft.ifft2(n_n0, s=self.etas[0].shape))

    ###################
    ## SIM FUNCTIONS ##
    ###################

    def run_one_step(self):
        """
        Runs one entire timestep for the amplitudes and the average density.
        """

        self.n0_old = self.n0.copy()
        self.n0 = self.n0_routine()

        if self.config["keepEtaConst"]:
            return

        n_etas = np.zeros(self.etas.shape, dtype=float)

        for eta_i in range(self.eta_count):
            n_etas[eta_i, :, :] = self.eta_routine(eta_i)

        self.etas = n_etas.copy()

    ##################
    ## IO FUNCTIONS ##
    ##################

    def reset_out_files(self, out_path: str):
        """
        Empties or creates the output files if they don't exist.

        Checks all :code:`out_{eta_i}.txt` files and the
        :code:`n0.txt` file.

        Args:
            out_path (str): directory of the output files.
        """

        for eta_i in range(self.eta_count):
            with open(f"{out_path}/out_{eta_i}.txt", "w") as f:
                f.write("")

        with open(f"{out_path}/n0.txt", "w") as f:
            f.write("")

    def write(self, out_path: str):
        """
        Writes the current content of
        :code:`etas` and :code:`n0` into the
        output files in :code:`out_{eta_i}.txt` and
        :code:`n0.txt`.
        Every line corresponds to one flattened entry.

        Args:
            out_path (str): Path where the out files are located.
        """

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
