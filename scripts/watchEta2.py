import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import json
from pympler.tracker import SummaryTracker
from pympler import muppy

import scipy

sys.setrecursionlimit(5000)
sns.set_theme()

sim_path = "/home/max/projects/apfc/data/tmp2"

with open(f"{sim_path}/config.json", "r") as f:
    config = json.load(f)

def read_eta(eta_path, dim):

    line = ""
    last_line = ""

    last_index = -1

    with open(eta_path, "r") as f:

        i = -1
        for next_line in f:

            if next_line == "\n" or next_line == "":
                break

            last_line = line
            line = next_line

            last_index = i
            i += 1

    splt = last_line.split(",")
    eta = np.array([float(s) for s in splt[:-1]])
    eta = eta.reshape((dim, dim))

    return eta, last_index

def get_surf_en(thetas, etas, config):

    global xm, ym

    G = np.array(config["G"])
    A = config["A"]

    int_sum = np.zeros(thetas.shape)

    h = np.diff(xm[0])[0]

    radii = np.sqrt(xm**2 + ym**2)

    for theta_i, theta in enumerate(thetas):

        rot_xm = np.cos(theta) * xm - np.sin(theta) * ym
        rot_ym = np.sin(theta) * xm + np.cos(theta) * ym

        rel_fields = np.logical_and(
            np.abs(rot_ym) < h / 2.,
            rot_xm >= 0.
        )

        for eta_i in range(G.shape[0]):

            rel_fields = np.logical_and(
                rel_fields,
                etas[eta_i] > 1e-5
            )

            xy = np.real(np.array([
                radii[rel_fields],
                etas[eta_i][rel_fields]
            ]))

            xy = xy[:,xy[0].argsort()]

            rad_min = np.min(xy[0])
            rad_max = np.max(xy[0])

            rad_lin = np.linspace(rad_min, rad_max, 100)
            xy_interpol = np.interp(rad_lin, xy[0], xy[1])

            xy = np.array([
                rad_lin,
                xy_interpol
            ])

            dx = np.diff(xy[0])[0]

            deta = np.gradient(xy[1], dx)
            d2eta = np.gradient(deta, dx)

            curv = d2eta * (1. + deta**2)**(-1.5)

            int_p1 = 8 * A * (G[eta_i, 0] * np.cos(theta) + G[eta_i, 1] * np.sin(theta))**2
            int_p1 += scipy.integrate.simpson(int_p1 * deta**2, xy[0])
            int_p1 += scipy.integrate.simpson(4 * A * curv**2 * deta**4, xy[0])

            int_sum[theta_i] += int_p1

        perc = (theta_i + 1) / thetas.shape[0] * 100
        sys.stdout.write(f"{perc:.4f}%\r")
        sys.stdout.flush()

    return int_sum

def get_stiffness(surf_en, thetas):
    dx = np.diff(thetas)[0]
    return surf_en + np.gradient(np.gradient(surf_en, dx), dx)

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)

fig = plt.figure()

axs = [
    [
        plt.subplot(221),
        plt.subplot(222, projection="polar"),
    ], [
        plt.subplot(223),
        plt.subplot(224, projection="polar")
    ]
]

for fig_arr in axs:
    for fig_ in fig_arr:
        fig_.set_aspect("equal")

div = make_axes_locatable(axs[0][0])
cax = div.append_axes("right", "5%", "5%")

first = True

def plot(frame):

    global fig, axs, div, cax
    global sim_path, config
    global xm, ym
    global first

    etas = []
    index = 0

    eta_paths = [
        f"{sim_path}/out_{i}.txt" for i in range(3)
    ]

    try:
        for eta_path in eta_paths:
            eta_, index = read_eta(eta_path, config["numPts"])
            etas.append(eta_)
    except Exception as e:
        print(e)
        return

    num_thetas = 1000

    thetas = np.linspace(0, 2. * np.pi, num_thetas, endpoint=True)

    thetas = np.hstack([
        thetas[num_thetas//2:],
        thetas,
        thetas[:num_thetas//2]
    ])

    surf_en = get_surf_en(thetas, etas, config)
    stiff = get_stiffness(surf_en, thetas)

    stiff = scipy.signal.savgol_filter(stiff, num_thetas // 10, 3)

    surf_en = surf_en[num_thetas//2:num_thetas+num_thetas//2]
    stiff = stiff[num_thetas//2:num_thetas+num_thetas//2]
    thetas = thetas[num_thetas//2:num_thetas+num_thetas//2]

    eta_sum = np.zeros(etas[0].shape)
    for eta_i in range(len(etas)):
        eta_sum += etas[eta_i] * etas[eta_i]

    min_ = np.min(eta_sum)
    max_ = np.max(eta_sum)

    axs[0][0].cla()
    cont = axs[0][0].contourf(xm, ym, eta_sum, 100)

    #colorbar creation causes memory leak
    if first:
        first = False
        cb = fig.colorbar(cont, cax=cax)
        cb.set_ticks([])

    axs[0][0].set_title(f"Sum of Etas\n{index}\nmax: {max_:.2e}\nmin: {min_:.2e}")

    axs[0][1].cla()
    axs[0][1].plot(thetas, surf_en)
    axs[0][1].set_title(f"Surface Energy\n{index}")
    axs[0][1].set_xticks([])
    axs[0][1].set_yticklabels([])

    axs[1][1].cla()
    axs[1][1].plot(thetas, stiff)
    axs[1][1].set_title(f"Stiffness\n{index}")
    axs[1][1].set_xticks([])
    axs[1][1].set_yticklabels([])

    plt.savefig("/home/max/projects/apfc/tmp/aaaa.png")

ani = FuncAnimation(plt.gcf(), plot, interval=1000, frames=100)
plt.show()
