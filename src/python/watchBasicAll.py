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

##############

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)


def plot_single(frame, xm, ym, eta_path, ax, config, plot_i, cax=None):

    eta_count = np.array(config["G"]).shape[0]

    etas = rw.read_all_etas_at_line(
        eta_path, plot_i, config["numPts"], config["numPts"], eta_count, float
    )

    eta_sum = np.zeros((config["numPts"], config["numPts"]), dtype=complex)
    for eta_i in range(eta_count):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])
    eta_sum = np.real(eta_sum).astype(float)

    ax.cla()
    cont = ax.contourf(xm, ym, eta_sum, 100)

    if cax is None:
        plt.colorbar(cont)
    else:
        plt.colorbar(cont, cax=cax)


index = 0


def plot_animate(frame, xm, ym, eta_path, ax, config, cax):

    global index

    eta_count = rw.count_lines(f"{eta_path}/out_0.txt")

    if index == eta_count:
        index = 0

    plot_single(frame, xm, ym, eta_path, ax, config, index, cax)

    index += 1


###############

fig = plt.figure()
ax = plt.subplot(111)
ax.set_aspect("equal")

###############

if args.animate:

    div = make_axes_locatable(ax)
    cax = div.append_axes("right", "5%", "5%")

    ani = FuncAnimation(
        plt.gcf(),
        plot_animate,
        interval=frame_time,
        fargs=(xm, ym, eta_path, ax, config, cax),
    )

else:

    plot_single(None, xm, ym, eta_path, ax, config, plot_i)

plt.show()
