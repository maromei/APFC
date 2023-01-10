import numpy as np
import scipy


def calc_surf_en_2d(xm, ym, etas, thetas, config):

    G = np.array(config["G"])
    A = config["A"]

    h = np.diff(xm[0])[0]
    radii = np.sqrt(xm**2 + ym**2)

    int_sum = np.zeros(thetas.shape)
    for theta_i, theta in enumerate(thetas):

        rot_xm = np.cos(theta) * xm - np.sin(theta) * ym
        rot_ym = np.sin(theta) * xm + np.cos(theta) * ym

        rel_fields = np.logical_and(np.abs(rot_ym) < h / 2.0, rot_xm >= 0.0)

        if rel_fields.nonzero()[0].shape[0] == 0:
            int_sum[theta_i] = None
            continue

        for eta_i in range(G.shape[0]):

            rel_fields = np.logical_and(rel_fields, etas[eta_i] > 1e-5)

            xy = np.real(np.array([radii[rel_fields], etas[eta_i][rel_fields]]))

            xy = xy[:, xy[0].argsort()]

            rad_min = np.min(xy[0])
            rad_max = np.max(xy[0])

            rad_lin = np.linspace(rad_min, rad_max, 100)
            xy_interpol = np.interp(rad_lin, xy[0], xy[1])

            dx = np.abs(np.abs(rad_lin[1] - np.abs(rad_lin[0])))

            int_sum[theta_i] += calc_single_surf_en_1d(
                rad_lin, xy_interpol, dx, theta, G[eta_i], A
            )

    return int_sum


def calc_single_surf_en_1d(x, y, dx, theta, G, A):

    deta = np.gradient(y, dx)
    d2eta = np.gradient(deta, dx)

    curv = d2eta * (1.0 + deta**2) ** (-1.5)

    integ = 8 * A * (G[0] * np.cos(theta) + G[1] * np.sin(theta)) ** 2
    integ += scipy.integrate.simpson(integ * deta**2, x)
    integ += scipy.integrate.simpson(4 * A * curv**2 * deta**4, x)

    return integ


def calc_surf_en_1d(xs, ys, dx, theta, G, A):

    integ = 0.0
    for eta_i in range(xs.shape[0]):
        integ += calc_single_surf_en_1d(xs[eta_i], ys[eta_i], dx, theta, G[eta_i], A)

    return integ


def calc_stiffness(surf_en, thetas):
    dx = np.diff(thetas)[0]
    return surf_en + np.gradient(np.gradient(surf_en, dx), dx)
