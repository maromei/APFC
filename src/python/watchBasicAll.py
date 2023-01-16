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
parser.add_argument("-s", "--split", action="store")
parser.add_argument("-f", "--frametime", action="store")
parser.add_argument("-pi", "--plotindex", action="store")
parser.add_argument("-a", "--animate", action="store_true")
parser.add_argument("-i", "--info", action="store_true")
parser.add_argument("-gif", "--savegif", action="store_true")
parser.add_argument("-dpi", "--dpi", action="store")

args = parser.parse_args()

################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

eta_path = sim_path
if args.split is not None:
    eta_path = f"{sim_path}/eta_files/{args.split}"

plot_i = args.plotindex
if plot_i is None:
    plot_i = -1
plot_i = int(plot_i)

frame_time = args.frametime
if frame_time is None:
    frame_time = 1000
frame_time = int(frame_time)

dpi = args.dpi
if dpi is None:
    dpi = 300
dpi = int(dpi)

##############

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)


def plot_single(frame, xm, ym, eta_path, axs, config, plot_i, args, cax=None):

    eta_count = np.array(config["G"]).shape[0]

    etas = rw.read_all_etas_at_line(
        eta_path, plot_i, config["numPts"], config["numPts"], eta_count, float
    )

    eta_sum = np.zeros((config["numPts"], config["numPts"]), dtype=complex)
    for eta_i in range(eta_count):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])
    eta_sum = np.real(eta_sum).astype(float)

    for ax in axs:
        ax.cla()

    axs[0].set_title(r"$\sum |\eta_i|^2$\vspace{1em}")
    cont = axs[0].contourf(xm, ym, eta_sum, 100)

    if args.info:
        txt = utils.build_sim_info_str(config, plot_i * config["writeEvery"])
        axs[1].axis("off")
        axs[1].text(
            0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center"
        )

    if cax is None:
        plt.colorbar(cont)
    else:
        plt.colorbar(cont, cax=cax)


index = 0


def plot_animate(frame, xm, ym, eta_path, axs, config, args, cax):

    global index

    eta_count = rw.count_lines(f"{eta_path}/out_0.txt")

    if index == eta_count:
        index = 0

    plot_single(frame, xm, ym, eta_path, axs, config, index, args, cax)

    index += 1


###############

if args.info:
    fig = plt.figure(figsize=(10, 5))
    axs = [plt.subplot(121), plt.subplot(122)]

    axs[1].axis("off")
    txt = utils.build_sim_info_str(config, 0)
    axs[1].text(0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center")

else:
    fig = plt.figure()
    axs = [
        plt.subplot(111),
    ]

for ax in axs:
    ax.set_aspect("equal")

axs[0].set_title(r"$\sum |\eta_i|^2$\vspace{1em}")

plt.tight_layout()

###############

if args.animate:

    div = make_axes_locatable(axs[0])
    cax = div.append_axes("right", "5%", "5%")

    if args.savegif:

        frames = rw.count_lines(f"{eta_path}/out_0.txt")

        ani = FuncAnimation(
            fig,
            plot_animate,
            interval=frame_time,
            fargs=(xm, ym, eta_path, axs, config, args, cax),
            frames=frames,
        )

        print("Saving GIF...")

        ani.save(f"{sim_path}/watch.gif", dpi=dpi)

        print("Displaying Animation...")

    ani = FuncAnimation(
        plt.gcf(),
        plot_animate,
        interval=frame_time,
        fargs=(xm, ym, eta_path, axs, config, args, cax),
    )
    plt.show()

else:

    if plot_i < 0:
        eta_count = rw.count_lines(f"{eta_path}/out_0.txt")
        plot_i = eta_count - np.abs(plot_i)
    plot_single(None, xm, ym, eta_path, axs, config, plot_i, args)

    plt.show()
