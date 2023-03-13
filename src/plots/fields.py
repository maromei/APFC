import numpy as np
import matplotlib.pyplot as plt


def plot_eta(etas: np.array, config: dict, axis: plt.Axes, cbar_cax=None):
    """
    Plots

    .. math::
        \sum\limits_m |\eta_m|^2

    Can handle both 1D and 2D amplitudes.

    Args:
        etas (np.array): The amplitudes with shape
            :code:`(eta_count, *(domain_shape))`
        config (dict): config object
        axis (plt.Axes): axis to plot it on
        cbar_cax: Either None or the colorbar axis. If None no colorbar will be
            generated. Only applies to 2D inputs.
    """

    is_1d = config["numPtsY"] <= 1
    eta_count = etas.shape[0]

    eta_sum = np.zeros(etas[0].shape, dtype=complex)
    for eta_i in range(eta_count):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])
    eta_sum = np.real(eta_sum)

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    title = r"$\sum\limits_m|\eta_m|^2$\vspace{1em}"

    if is_1d:

        plot_1d(x, eta_sum.flatten(), title, axis)

    else:

        y = np.linspace(-config["xlim"], config["xlim"], config["numPtsY"], axis)
        xm, ym = np.meshgrid(x, y)
        plot_2d(xm, ym, eta_sum, title, axis, cbar_cax)


def plot_n0(n0: np.array, config: dict, axis: plt.Axes, cbar_cax=None):
    """
    Plots the average density :math:`n_0`. Can handle both 1D and 2D
    input.s

    Args:
        n0 (np.array): average density
        config (dict): config object
        axis (plt.Axes): axis to plot on
        cbar_cax: Either None or the colorbar axis. If None no colorbar will be
            generated. Only applies to 2D inputs.
    """

    is_1d = config["numPtsY"] <= 1

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])
    title = r"$n_0$\vspace{1em}"

    if is_1d:

        plot_1d(x, n0.flatten(), title, axis)

    else:

        y = np.linspace(-config["xlim"], config["xlim"], config["numPtsY"], axis)
        xm, ym = np.meshgrid(x, y)
        plot_2d(xm, ym, n0, title, axis, cbar_cax)


def plot_1d(x: np.array, y: np.array, title: str, axis: plt.Axes):
    """
    Plots whatever x and y is with the given title.

    Args:
        x (np.array):
        y (np.array):
        title (str):
        axis (plt.Axes): axis to plot on
    """

    axis.plot(x, y)
    axis.set_title(title)


def plot_2d(
    xm: np.array, ym: np.array, zm: np.array, title: str, axis: plt.Axes, cbar_cax=None
):
    """
    Plots the input as a contourf with 100 colors.

    Args:
        xm (np.array):
        ym (np.array):
        zm (np.array):
        title (str):
        axis (plt.Axes): axis to plot on
        cbar_cax: Either None or the colorbar axis. If None no colorbar will be
            generated.
    """

    cont = axis.contourf(xm, ym, zm, 100)
    axis.set_title(title)

    if cbar_cax is not None:
        plt.colorbar(cont, cax=cbar_cax)
