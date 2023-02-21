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
parser.add_argument("-save", "--save", action="store_true")
parser.add_argument("-dpi", "--dpi", action="store")
parser.add_argument("-pe", "--plot-every", action="store", default=1, type=int)

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


def plot_single(frame, x, eta_path, axs, config, plot_i, args):

    eta_count = np.array(config["G"]).shape[0]

    etas = rw.read_all_etas_at_line(
        eta_path, plot_i, config["numPts"], 1, eta_count, float
    )

    n0 = rw.read_eta_at_line(f"{eta_path}/n0.txt", plot_i, config["numPts"], 1, float)

    eta_sum = np.zeros(config["numPts"], dtype=complex)
    for eta_i in range(eta_count):
        eta_sum += etas[eta_i].flatten() * np.conj(etas[eta_i].flatten())
    eta_sum = np.real(eta_sum).astype(float)

    for ax in axs:
        ax.cla()

    axs[0].set_title(r"$\sum |\eta_i|^2$\vspace{1em}")
    axs[0].plot(x, eta_sum.flatten())

    axs[1].set_title(r"$n_0$\vspace{1em}")
    axs[1].plot(x, n0.flatten())

    if args.info:

        theta = None
        if args.split is not None:
            theta = float(args.split)

        txt = utils.build_sim_info_str(
            config, plot_i * config["writeEvery"], theta=theta, is_1d=True
        )
        axs[2].axis("off")
        axs[2].text(
            0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center"
        )


index = 0


def plot_animate(frame, x, eta_path, axs, config, args):

    global index

    eta_count = rw.count_lines(f"{eta_path}/out_0.txt")

    if index == eta_count:
        index = 0

    while index % args.plot_every != 0:
        index += 1

    plot_single(frame, x, eta_path, axs, config, index, args)

    index += 1


###############

if args.info:
    fig = plt.figure(figsize=(10, 8))
    axs = [plt.subplot(221), plt.subplot(222), plt.subplot(223)]

    axs[2].axis("off")
    txt = utils.build_sim_info_str(config, 0)
    axs[2].text(0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center")
    axs[2].set_aspect("equal")

else:
    fig = plt.figure()
    axs = [
        plt.subplot(111),
    ]

axs[0].set_title(r"$\sum |\eta_i|^2$\vspace{1em}")
axs[1].set_title(r"$n_0$\vspace{1em}")

plt.tight_layout()

###############

if args.animate:

    frames = rw.count_lines(f"{eta_path}/out_0.txt")

    ani = FuncAnimation(
        fig,
        plot_animate,
        interval=frame_time,
        fargs=(x, eta_path, axs, config, args),
        frames=frames,
    )

    if args.save:
        ani.save(f"{sim_path}/watch.gif", dpi=dpi)

    plt.show()

else:

    if plot_i < 0:
        eta_count = rw.count_lines(f"{eta_path}/out_0.txt")
        plot_i = eta_count - np.abs(plot_i)
    plot_single(None, x, eta_path, axs, config, plot_i, args)

    if args.save:
        plt.savefig(f"{sim_path}/watch_{plot_i}.png", dpi=dpi)

    plt.show()
