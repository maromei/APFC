import numpy as np
import scipy

from .initialize import tanhmin

from . import params


def calc_single_surf_en_1d(
    x: np.array, eta: np.array, theta: float, G: np.array, A: float
) -> float:
    """
    Calculates one single summand of equation
    :eq:`eqn:surf_en_calc_1d`.

    Args:
        x (np.array): x-values with shape :code:`(n, 1)` or :code:`(n,)`
        eta (np.array): a single amplitude with shape
            :code:`(n, 1)` or :code:`(n,)`
        theta (float): The rotation of the reciprical vectors
        G (np.array): The unrotated resiprical vectors with shape
            :code:`(2, 1)` or :code:`(2,)`.
        A (float): parameter of equation :eq:`eqn:apfc_flow_constants`

    Returns:
        float: Integral result
    """

    # fmt: off
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # fmt: on
    G = rot.dot(G)

    dx = np.abs(x[0] - x[1])

    deta = np.gradient(eta, dx)
    d2eta = np.gradient(deta, dx)
    d3eta = np.gradient(d2eta, dx)

    gsq = (G[0] * np.cos(theta) + G[1] * np.sin(theta)) ** 2

    integ = 8 * gsq * deta**2
    integ += 4 * d2eta**2
    integ -= 2 * deta * d3eta
    integ = A * scipy.integrate.simpson(integ, x)

    return integ


def calc_surf_en_1d(etas: np.array, n0, config, theta: float) -> float:
    """
    Calculates the 1d surface energy according to equation
    :eq:`eqn:surf_en_calc_1d`.

    Args:
        x (np.array): x-values with shape
            :code:`(n,)`.
        etas (np.array): Ampltiudes with shape :code:`(eta_count, n)`
        theta (float): The rotation of the reciprical vectors
        G (np.array): The unrotated resiprical vectors with shape
            :code:`(eta_count, 2)` or :code:`(eta_count, 2)`.
        A (float): parameter of equation :eq:`eqn:apfc_flow_constants`

    Returns:
        float: surface energy
    """

    config = config.copy()

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    G = np.array(config["G"])
    A = config["Bx"]

    integ = 0.0
    for eta_i in range(etas.shape[0]):
        integ += calc_single_surf_en_1d(x, etas[eta_i], theta, G[eta_i], A)

    return integ


def calc_surf_en_1d2(etas: np.array, n0, config_, theta: float) -> float:

    config = config_.copy()

    G = np.array(config["G"])
    # fmt: off
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # fmt: on
    for eta_i in range(G.shape[0]):
        G[eta_i] = rot.dot(G[eta_i])
    config["G"] = G.tolist()

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    dx = np.abs(np.diff(x)[0])

    i_eta = etas[0][int(etas[0].shape[0] / 2) :]
    i_x = x[int(etas[0].shape[0] / 2) :]
    interface_width = get_interface_width(i_x, i_eta.flatten())

    F = energy_functional_1d(etas, n0, config, theta)
    chem_pot = get_chemical_potential(etas, n0, config)

    _, Vl = get_phase_volumes(etas[0], dx)

    if config["simType"] == "n0":
        _, nl = get_phase_volumes(n0, dx)
    else:
        nl = n0

    liq_etas = []
    for eta_i in range(etas.shape[0]):
        _, liq = get_phase_eq_values(etas[eta_i])
        liq_etas.append(liq)
    liq_etas = np.array(liq_etas)

    ret = F - scipy.integrate.simpson(chem_pot, x)
    fl = sub_energy_functional_1d(etas, n0, config, theta)[0]
    mul = get_chemical_potential(liq_etas, nl, config)
    ret -= Vl * (fl - mul * nl)

    return ret / interface_width


def get_chemical_potential(etas, n0, config):

    phi_ = phi(etas)

    p = np.ones(etas[0].shape)
    for eta_i in range(etas.shape[0]):
        p *= etas[eta_i]
    p += np.conj(p)
    p *= 2

    ret = (config["dB0"] + config["Bx"]) * n0
    ret += 3 * config["v"] * phi_ * n0
    ret += 3 * config["v"] * p
    ret -= phi_ * config["t"]
    ret -= config["t"] * n0**2
    ret += config["v"] * n0**3

    return ret


def triangular_one_mode_func(etas, n0, config):

    c = params.C(config, n0)

    ret = np.ones(etas[0].shape)
    for eta_i in range(etas.shape[0]):
        ret *= etas[eta_i]

    ret += np.conj(ret)

    return 2 * c * ret


def energy_functional_1d(etas, n0, config, theta):

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])

    sub_fun = sub_energy_functional_1d(etas, n0, config, theta)
    ret = scipy.integrate.simpson(sub_fun, x)

    return ret


def sub_energy_functional_1d(etas, n0, config, theta):

    # fmt: off
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # fmt: on

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    G = np.array(config["G"])
    for eta_i in range(G.shape[0]):
        G[eta_i] = rot.dot(G[eta_i])

    a = params.A(config)
    b = params.B(config, n0)
    d = params.D(config)
    e = params.E(config, n0)

    phi_ = phi(etas)
    tri_f = triangular_one_mode_func(etas, n0, config)

    sum_ = np.zeros(etas[0].shape)
    for eta_i in range(etas.shape[0]):

        deta = np.gradient(etas[eta_i].flatten(), x)
        d2eta = np.gradient(deta, x)

        op = d2eta**2 + 4 * G[eta_i].dot(G[eta_i])

        sum_ += a * op - 3 * d / 2 * etas[eta_i] ** 4

    ret = b / 2 * phi_
    ret += 3 * d / 4 * phi_**2
    ret += op + tri_f + e

    return ret


def calc_stiffness(surf_en: np.array, thetas: np.array) -> np.array:
    """
    Calculates the stiffness accroding to equation
    :eq:`eqn:stiffness`.

    Args:
        surf_en (np.array): Surface energy per angle
        thetas (np.array): angle

    Returns:
        np.array: stiffness per angle
    """

    dx = np.diff(thetas)[0]
    return surf_en + np.gradient(np.gradient(surf_en, dx), dx)


def get_phase_eq_values(arr: np.array) -> tuple[float, float]:

    max_val = np.max(arr)
    min_val = np.min(arr)

    return max_val, min_val


def get_phase_volumes(arr: np.array, dx=float, dy: float = 1.0) -> tuple[float, float]:

    max_val = np.max(arr)
    is_liq = arr < max_val / 2

    step_area = dx * dy
    total_area = arr.flatten().shape[0] * step_area

    liq_area = np.sum(is_liq) * step_area
    sol_area = total_area - liq_area

    return sol_area, liq_area


def get_interface_width(x: np.array, y: np.array) -> float:

    tanhfit = lambda x, r, eps: tanhmin(x - r, eps)
    popt, pcov = scipy.optimize.curve_fit(tanhfit, x, y / np.max(y))

    if np.any(pcov > 1e-1):
        print("WARNING:")
        print("Fitting interface width resulted in large variance!", pcov)

    return popt[1]


def phi(etas):

    ret = np.zeros(etas[0].shape, dtype=float)
    for eta_i in range(etas.shape[0]):
        ret += etas[eta_i] * np.conj(etas[eta_i])
    return 2 * ret
