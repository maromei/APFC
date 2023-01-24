import numpy as np
import scipy
import copy


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


def calc_line_surf_en(
    xm,
    ym,
    config_,
    etas,
    theta=0.0,
    rot_g=True,
    average=False,
    integ2d=False,
    use_pos=True,
):

    config = copy.deepcopy(config_)
    G = np.array(config["G"])

    if use_pos:

        pos_con = xm[0, :] >= 0.0
        xm = xm[:, pos_con]
        ym = ym[:, pos_con]

        etas_ = np.zeros((G.shape[0], xm.shape[0], xm.shape[1]))
        for eta_i in range(G.shape[0]):
            etas_[eta_i] = etas[eta_i][:, pos_con]
        etas = etas_

    x_row = xm[0, :]
    dx = np.diff(np.abs(x_row))[0]
    xs = np.vstack([x_row for _ in range(G.shape[0])])

    if rot_g:

        # fmt: off
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # fmt: on

        for eta_i in range(G.shape[0]):
            G[eta_i] = rot.dot(G[eta_i])
        config["G"] = G.tolist()

    if average:

        ys = np.vstack([np.average(etas[eta_i], axis=0) for eta_i in range(G.shape[0])])

        surf_en = calc_surf_en_1d(xs, ys, dx, theta, G, config["A"])

    elif integ2d:

        integ = np.zeros(xm.shape[0])
        y_col = ym[:, 0]

        for i in range(integ.shape[0]):

            ys = np.vstack([etas[eta_i][i, :] for eta_i in range(G.shape[0])])

            integ[i] = calc_surf_en_1d(xs, ys, dx, theta, G, config["A"])

        surf_en = scipy.integrate.simpson(integ, y_col)

    else:

        y_middle = config["numPts"] // 2
        ys = np.vstack([etas[eta_i][y_middle, :] for eta_i in range(G.shape[0])])

        surf_en = calc_surf_en_1d(xs, ys, dx, theta, G, config["A"])

    return surf_en


def theo_surf_en(thetas, eps, gamma_0):
    return gamma_0 * (1.0 + eps * np.cos(6.0 * thetas))


def theo_surf_en_sec_der(thetas, eps, gamma_0):
    return -gamma_0 * eps * 36.0 * np.cos(6.0 * thetas)


def theo_surf_en_der(thetas, eps, gamma_0):
    return -gamma_0 * eps * 6.0 * np.sin(6.0 * thetas)


def wulf_shape(thetas, eps, gamma):

    surf = theo_surf_en(thetas, eps, gamma)
    surf_der = theo_surf_en_der(thetas, eps, gamma)

    x = surf * np.cos(thetas) - surf_der * np.sin(thetas)
    y = surf * np.sin(thetas) + surf_der * np.cos(thetas)

    return x, y


def fit_surf_en(thetas, line_en):

    popt, pcov = scipy.optimize.curve_fit(theo_surf_en, thetas, line_en)
    return popt


def calc_stiffness(surf_en, thetas):
    dx = np.diff(thetas)[0]
    return surf_en + np.gradient(np.gradient(surf_en, dx), dx)


def calc_stiffness_fit(surf, thetas, eps, gamma0):

    return surf + theo_surf_en_sec_der(thetas, eps, gamma0)
