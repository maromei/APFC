import numpy as np
import scipy


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


def calc_surf_en_1d(
    x: np.array, etas: np.array, theta: float, G: np.array, A: float
) -> float:
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

    integ = 0.0
    for eta_i in range(etas.shape[0]):
        integ += calc_single_surf_en_1d(x, etas[eta_i], theta, G[eta_i], A)

    return integ


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
