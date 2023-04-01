import numpy as np
import scipy

from manage import utils

from .initialize import tanhmin
from . import params


def calc_surf_en_1d(etas, n0, config, theta, div_interface_width=True):

    config = config.copy()
    eta_count = etas.shape[0]

    is_n0_sim = config["simType"] == "n0"

    ##########################
    ### get positive range ###
    ##########################

    x_full = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    x, etas = utils.get_positive_range(x_full, etas, True)

    dx = np.abs(x_full[1] - x_full[0])

    if is_n0_sim:
        _, n0 = utils.get_positive_range(x_full, n0)

    ################
    ### rotate G ###
    ################

    G = np.array(config["G"])
    G_rot = G.copy()

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    for eta_i in range(eta_count):
        G_rot[eta_i] = rot.dot(G[eta_i])

    ###############
    ### eta sum ###
    ###############

    eta_sum = np.zeros(etas[0].shape)
    for eta_i in range(eta_count):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])

    interface_width = 1
    if div_interface_width:
        interface_width = get_interface_width(x, eta_sum, True)
        if interface_width is None:
            return 0.0

    ########################
    ### integrand values ###
    ########################

    _, n0_liq = get_phase_eq_values(n0)

    f = sub_energy_functional_1d(etas, n0, dx, config, G_rot)
    mu = get_chemical_potential(etas, n0, config)

    # this array is needed because the sub_energy_functional calculates
    # gradients. It is a waste of space, but bettern than rewriting the
    # fucntion. It is in the shape of all etas, so the output of the
    # subenergy functional fits to the f shape and the `ret` calculation
    # below works seemlessly.
    full_etas_liq = np.zeros(etas.shape)
    f_liq = sub_energy_functional_1d(full_etas_liq, n0_liq, dx, config, G_rot)

    #################
    ### integrate ###
    #################

    ret = f - mu * n0 - f_liq + mu * n0_liq
    ret = scipy.integrate.simpson(ret, x)

    if div_interface_width:
        ret = ret / interface_width

    return ret


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


def energy_functional_1d(etas, n0, x, config, theta):

    sub_fun = sub_energy_functional_1d(etas, n0, x, config, theta)
    ret = scipy.integrate.simpson(sub_fun, x)

    return ret


def sub_energy_functional_1d(etas, n0, dx, config, G):

    a = params.A(config)
    b = params.B(config, n0)
    d = params.D(config)
    e = params.E(config, n0)

    phi_ = phi(etas)
    tri_f = triangular_one_mode_func(etas, n0, config)

    sum_ = np.zeros(etas[0].shape)
    for eta_i in range(etas.shape[0]):

        deta = np.gradient(etas[eta_i].flatten(), dx)
        d2eta = np.gradient(deta, dx)

        G_elem = G[eta_i, 0] ** 2 + G[eta_i, 1] ** 2

        op = d2eta**2 + G_elem * deta**2

        sum_ += a * op - 3 * d / 2 * etas[eta_i] ** 4

    ret = b / 2 * phi_
    ret += 3 * d / 4 * phi_**2
    ret += sum_ + tri_f + e

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

    o_len = surf_en.shape[0]
    en = utils.fill(surf_en, 3, False)
    dx = np.diff(thetas)[0]

    stiff = en + np.gradient(np.gradient(en, dx), dx)

    stiff = stiff[o_len - 1 : 2 * o_len - 1]

    return stiff


def get_phase_eq_values(arr: np.array) -> tuple[float, float]:

    max_val = np.max(arr)
    min_val = np.min(arr)

    return max_val, min_val


def get_phase_volumes(arr: np.array, dx=float, dy: float = 1.0) -> tuple[float, float]:

    max_val = np.max(arr)
    min_val = np.min(arr)
    threshhold = (max_val - min_val) / 2

    is_liq = arr < threshhold

    step_area = dx * dy
    total_area = arr.flatten().shape[0] * step_area

    liq_area = np.sum(is_liq) * step_area
    sol_area = total_area - liq_area

    return sol_area, liq_area


def get_interface_width(x: np.array, y: np.array, silent: bool = True) -> float:

    return fit_to_tanhmin(x, y, silent)[1]


def fit_to_tanhmin(x: np.array, y: np.array, silent: bool = True) -> float:

    tanhfit = lambda x, r, eps: tanhmin(x - r, eps)

    y_fit = y - np.min(y)
    y_fit = y_fit / np.max(y)

    dx = np.abs(x[0] - x[1])

    ### make first general estimate for fit ###

    dy = np.abs(np.gradient(y_fit, dx))
    dy = dy / np.max(dy)

    dyoverthresh = dy > 0.2
    initIntWidth = np.sum(dyoverthresh) * dx

    if initIntWidth == 0:
        return 0.0, 0.0

    sol_area = np.median(x[np.where(dyoverthresh)])

    ###  ###

    popt, pcov = scipy.optimize.curve_fit(
        tanhfit, x, y_fit, p0=[sol_area, initIntWidth]
    )

    if np.any(pcov > 1) or np.any(np.isnan(pcov)):
        if not silent:
            print("WARNING:")
            print("Fitting interface width resulted in large variance!", pcov)
        return None, None

    return popt


def phi(etas):

    ret = np.zeros(etas[0].shape, dtype=float)
    for eta_i in range(etas.shape[0]):
        ret += etas[eta_i] * np.conj(etas[eta_i])
    return 2 * ret
