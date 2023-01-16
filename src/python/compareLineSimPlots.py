import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sim import read_write as rw
from sim import utils

matplotlib.use("TkAgg")
sns.set_theme()
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(prog="BasicEtaWatcher", description="Watch eta")

parser.add_argument("sim_path")
parser.add_argument("-f", "--frametime", action="store", type=int, default=1000)
parser.add_argument("-save", "--save", action="store_true")
parser.add_argument("-dpi", "--dpi", action="store", type=int, default=300)
parser.add_argument("-th", "--thetas", action="store")

args = parser.parse_args()

theta_strs = args.thetas.split(",")
theta_len = len(theta_strs)

if not (2 <= theta_len <= 5):
    raise ValueError("Only 2-5 thetas are supported")

################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

################
## BUILD VARS ##
################

layout_map = {2: (2, 2), 3: (2, 2), 4: (3, 2), 5: (3, 2)}
layout = layout_map[theta_len]
layout_str = f"{layout[0]}{layout[1]}"

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)


def gets_plot(row_i, col_i, theta_len):

    i = col_i * layout[0] + row_i
    return i < theta_len


fig = plt.figure()

axs = []
for row in range(layout[0]):
    axs.append([])
    for col in range(layout[1]):

        i = col * layout[0] + row

        axs[row].append(plt.subplot(int(f"{layout_str}{i+1}")))

cax = []
for ax_row_i, ax_row in enumerate(axs):
    cax.append([])
    for col_i, ax in enumerate(ax_row):

        ax.set_aspect("equal")
        ax.set_title("$\\theta=0.0000$")

        if not gets_plot(ax_row_i, col_i, theta_len):
            continue

        div = make_axes_locatable(ax)
        cax[ax_row_i].append(div.append_axes("right", "5%", "5%"))

line_count = rw.count_lines(f"{sim_path}/eta_files/{theta_strs[0]}/out_0.txt")

index = 0


def plot(frame, theta_strs, sim_path, xm, ym, config, axs, cax, theta_len, line_count):

    global index

    if index == line_count:
        index = 0

    # reset all plots
    for axs_row in axs:
        for ax in axs_row:
            ax.cla()

    for row_i, row in enumerate(axs):
        for col_i, ax in enumerate(row):

            if not gets_plot(row_i, col_i, theta_len):

                ax.axis("off")

                continue

            i = col_i * len(axs) + row_i

            etas = rw.read_all_etas_at_line(
                f"{sim_path}/eta_files/{theta_strs[i]}",
                index,
                config["numPts"],
                config["numPts"],
                len(config["G"]),
                float,
            )

            eta_sum = np.zeros((config["numPts"], config["numPts"]), dtype=complex)
            for eta_i in range(len(config["G"])):
                eta_sum += etas[eta_i] * np.conj(etas[eta_i])
            eta_sum = np.real(eta_sum).astype(float)

            cont = ax.contourf(xm, ym, eta_sum, 100)
            plt.colorbar(cont, cax=cax[row_i][col_i])
            ax.set_title(f"$\\theta = {theta_strs[i]}$")

    txt = utils.build_sim_info_str(config, index)

    axs[-1][-1].text(
        0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center"
    )

    index += 1


ani = FuncAnimation(
    fig,
    plot,
    interval=args.frametime,
    fargs=(theta_strs, sim_path, xm, ym, config, axs, cax, theta_len, line_count),
    frames=line_count,
)

if args.save:
    ani.save(f"{sim_path}/compLineSimPlt{'_'.join(theta_strs)}.gif", dpi=args.dpi)

plt.show()
