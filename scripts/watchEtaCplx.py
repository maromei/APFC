import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import json

sys.setrecursionlimit(5000)
sns.set_theme()

sim_path = "/home/max/projects/apfc/data/stencil/1.59"

with open(f"{sim_path}/config.json", "r") as f:
    config = json.load(f)

def read_complex(line, dim1, dim2):

    split = line.split("),(")
    split = [s.replace("(", "").replace(")", "") for s in split[:]]
    arr = []
    for ss in split:
        s = ss.split(",")
        arr.append(complex(float(s[0]), float(s[1])))
    arr = np.array(arr)
    arr = arr.reshape((dim1, dim2))

    return arr

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

    eta = read_complex(last_line, dim, dim)

    return eta, last_index

def get_surf_en(thetas, etas, config):

    global xm, ym

    G = np.array(config["G"])
    A = config["A"]

    # grid spacing to get tolerance for
    # selecting relevant fields
    # equal spacing assumed
    h = np.diff(xm[0])[0]
    h_r = lambda r, h: 2 * np.arcsin(h / (2. * r))

    # get relevant angles
    angles = np.arctan2(ym, xm)
    ang_small_0 = angles < 0 # for -y --> negative angles --> correct
    angles[ang_small_0] = 2. * np.pi - np.abs(angles[ang_small_0])

    radii = np.sqrt(xm**2 + ym**2)

    int_sum = np.zeros(thetas.shape)

    for theta_i, theta in enumerate(thetas):

        rel_angles = np.logical_and(
            angles <= theta + h,
            angles >= theta - h
        )

        for eta_i in range(G.shape[0]):

            xy = np.real(np.array([
                radii[rel_angles],
                etas[eta_i][rel_angles]
            ]))

            xy = xy[:,xy[0].argsort()]

            deta = np.gradient(xy[1])
            d2eta = np.gradient(deta)

            curv = d2eta * (1. + deta**2)**(-1.5)

            int_p1 = 8 * A * (G[eta_i, 0] * np.cos(theta) + G[eta_i, 1] * np.sin(theta))**2
            int_p1 = np.trapz(int_p1 * deta**2, xy[0])

            int_p2 = np.trapz(4 * A * curv**2 * deta**4)

            int_sum[theta_i] += int_p1 + int_p2

    return int_sum

def get_surf_en2(thetas, etas, config):

    global xm, ym

    G = np.array(config["G"])
    A = config["A"]

    int_sum = np.zeros(thetas.shape)

    h = np.diff(xm[0])[0]

    for theta_i, theta in enumerate(thetas):

        rot_xm = np.cos(theta) * xm + np.sin(theta) * ym
        rot_ym = np.cos(theta) * ym - np.sin(theta) * xm

        rel_fields = np.logical_and(
            rot_xm <= h / 2.,
            rot_xm >= - h / 2.
        )

        radii = np.sqrt(rot_xm**2 + rot_ym**2)

        for eta_i in range(G.shape[0]):

            xy = np.real(np.array([
                radii[rel_fields],
                etas[eta_i][rel_fields]
            ]))

            xy = xy[:,xy[0].argsort()]

            deta = np.gradient(xy[1])
            d2eta = np.gradient(deta)

            curv = d2eta * (1. + deta**2)**(-1.5)

            int_p1 = 8 * A * (G[eta_i, 0] * np.cos(theta) + G[eta_i, 1] * np.sin(theta))**2
            int_p1 = np.trapz(int_p1 * deta**2, xy[0])

            int_p2 = np.trapz(4 * A * curv**2 * deta**4)

            int_sum[theta_i] += int_p1 + int_p2

    return int_sum

def get_stiffness(surf_en):
    return surf_en + np.gradient(np.gradient(surf_en))

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
    except:
        return

    #thetas = np.linspace(0, 2. * np.pi, 1000)
    #surf_en = get_surf_en(thetas, etas, config)
    #stiff = get_stiffness(surf_en)

    eta_sum = np.zeros(etas[0].shape, dtype=complex)
    for eta_i in range(len(etas)):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])

    min_ = np.min(np.real(np.abs(eta_sum)))
    max_ = np.max(np.real(np.abs(eta_sum)))

    eta_sum = np.real(np.abs(eta_sum))

    axs[0][0].cla()
    cont = axs[0][0].contourf(xm, ym, eta_sum, 100)

    #colorbar creation causes memory leak
    if first:
        first = False
        cb = fig.colorbar(cont, cax=cax)
        cb.set_ticks([])

    axs[0][0].set_title(f"Sum of Etas\n{index}, min: {min_:.2f}, max_: {max_:.2f}")

    #axs[0][1].cla()
    #axs[0][1].plot(thetas, surf_en)
    #axs[0][1].set_title(f"Surface Energy\n{index}")
    #axs[0][1].set_xticks([])
    #axs[0][1].set_yticklabels([])
#
    #axs[1][1].cla()
    #axs[1][1].plot(thetas, stiff)
    #axs[1][1].set_title(f"Stiffness\n{index}")
    #axs[1][1].set_xticks([])
    #axs[1][1].set_yticklabels([])

ani = FuncAnimation(plt.gcf(), plot, interval=1000, frames=100)
plt.show()
