import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from sim import calculations as calc
from sim import read_write as rw
from sim import utils

matplotlib.use("TkAgg")

import sys

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_theme()
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

sys.setrecursionlimit(5000)

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="FFTWatchGrainSim", description="FFT Watcher for simple grain sim"
)

parser.add_argument("sim_path")
parser.add_argument("-th", "--thetacount", action="store")
parser.add_argument("-otp", "--onetimeplot", action="store_true")
parser.add_argument("-ft", "--frametime", action="store")
parser.add_argument("-isc", "--ignoresurcalc", action="store_true")

args = parser.parse_args()

one_time_plot = args.onetimeplot

theta_count = args.thetacount
theta_count = 1000 if theta_count is None else int(theta_count)

frame_time = args.frametime
frame_time = 1000 if frame_time is None else int(frame_time)

################
## GET CONFIG ##
################

config_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{config_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

################
## BUILD VARS ##
################

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)

G = np.array(config["G"])

thetas = np.linspace(0, 2.0 * np.pi, theta_count)
thetas_calc = np.hstack(
    [thetas[theta_count // 2 :], thetas, thetas[: theta_count // 2]]
)

eta_paths = [f"{args.sim_path}/out_{i}.txt" for i in range(G.shape[0])]

####################
## PLOT FUNCTIONS ##
####################


def plot(frame):

    #############
    ## GLOBALS ##
    #############

    global axs, fig, cont, cax
    global eta_paths, one_time_plot
    global xm, ym, thetas, thetas_calc
    global config
    global first
    global args

    ###############
    ## READ ETAS ##
    ###############

    etas = []

    index = 0
    try:
        for eta_path in eta_paths:
            eta_, index = rw.read_eta_last_file(
                eta_path, config["numPts"], config["numPts"], dtype=float
            )
            etas.append(eta_)
    except Exception as e:
        print(e)
        return

    ##################
    ## CALCULATIONS ##
    ##################

    eta_sum = np.zeros(etas[0].shape, dtype=complex)
    for eta_i in range(len(etas)):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])
    eta_sum = np.real(eta_sum).astype(float)

    min_ = np.min(eta_sum)
    max_ = np.max(eta_sum)

    step_index = config["writeEvery"] * index

    if args.ignoresurcalc:
        surf_en = np.zeros(thetas.shape)
        stiff = np.zeros(thetas.shape)
    else:
        surf_en = calc.calc_surf_en_2d(xm, ym, etas, thetas_calc, config)
        stiff = calc.calc_stiffness(surf_en, thetas_calc)

        stiff = scipy.signal.savgol_filter(stiff, theta_count // 10, 3)

        surf_en = surf_en[theta_count // 2 : theta_count + theta_count // 2]
        stiff = stiff[theta_count // 2 : theta_count + theta_count // 2]
        thetas = thetas_calc[theta_count // 2 : theta_count + theta_count // 2]

    ###########
    ## PLOTS ##
    ###########

    ####################
    ## clear all axes ##
    ####################

    for row in axs:
        for ax in row:
            ax.cla()

    ###############
    ## text axis ##
    ###############

    txt = f"""\
        \\begin{{center}}
        sim iteration: {step_index} \\vspace{{0.5em}}
        $B_x = {config['Bx']:.4f}, n_0 = {config['n0']:.4f}$
        $v = {config['v']:.4f}, t = {config['t']:.4f}$
        $\\Delta B_0 = {config['dB0']:.4f}$
        $\\mathrm{{d}}t = {config['dt']:.4f}$ \\vspace{{0.5em}}
        initial Radius: {config['initRadius']:.4f}
        initial Eta in solid: {config['initEta']:.4f}
        interface width: {config['interfaceWidth']:.4f}
        domain: $[-{config['xlim']}, {config['xlim']}]^2$
        points: {config['numPts']} x {config['numPts']}
        \\end{{center}}
    """

    txt = "".join(map(str.lstrip, txt.splitlines(1)))

    axs[1][0].axis("off")
    axs[1][0].text(
        0.5, 0.5, txt, verticalalignment="center", horizontalalignment="center"
    )

    ###############
    ## main plot ##
    ###############

    txt = f"\\vspace{{-0.5em}}\\begin{{center}}$\\sum\\limits_m |\\eta_m|^2$\nmax: {max_:.2e} min: {min_:.2e}\\end{{center}}\\vspace{{-1em}}"

    if one_time_plot:
        txt = f"\\vspace{{-0.5em}}\\begin{{center}}$\\sum\\limits_m |\\eta_m|^2$\\end{{center}}\\vspace{{-1em}}"

    axs[0][0].set_title(txt)
    cont = axs[0][0].contourf(xm, ym, eta_sum, 100)

    ############################
    ## SURFACE EN / STIFFNESS ##
    ############################

    if not args.ignoresurcalc:

        axs[0][1].plot(thetas, surf_en)
        axs[0][1].set_title(f"Surface Energy")
        axs[0][1].set_xticks([])
        axs[0][1].set_yticklabels([])

        axs[1][1].plot(thetas, stiff)
        axs[1][1].set_title(f"Stiffness")
        axs[1][1].set_xticks([])
        axs[1][1].set_yticklabels([])

    #########################
    ## Deal with first use ##
    #########################
    # this has to be last

    if first:
        first = False
        cb = fig.colorbar(cont, cax=cax)
        if not one_time_plot:
            cb.set_ticks([])


################
## PLOT SETUP ##
################

# fmt: off

if args.ignoresurcalc:

    fig = plt.figure(figsize=(10, 5))

    axs = [
        [plt.subplot(121),],
        [plt.subplot(122),]
    ]

else:

    fig = plt.figure(figsize=(10, 8))

    axs = [
        [
            plt.subplot(221),
            plt.subplot(222, projection="polar"),
        ],
        [
            plt.subplot(223),
            plt.subplot(224, projection="polar")
        ],
    ]

# fmt: on

for fig_arr in axs:
    for fig_ in fig_arr:
        fig_.set_aspect("equal")

axs[0][0].set_title("a\nb\nc\n\\vspace{-0.5em}")

div = make_axes_locatable(axs[0][0])
cax = div.append_axes("right", "5%", "5%")

first = True

plt.tight_layout()

##############
## RUN PLOT ##
##############

if one_time_plot:

    plot("")

else:
    ani = FuncAnimation(plt.gcf(), plot, interval=frame_time, frames=100)

plt.show()
