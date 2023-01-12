import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sim import calculations as calc
from sim import utils

matplotlib.use("TkAgg")
sns.set_theme()
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="LineAnimSurfEn", description="Animate the surf en for line sim"
)

parser.add_argument("sim_path")
parser.add_argument("-ft", "--frametime", action="store")
parser.add_argument("-f", "--fill", action="store_true")
parser.add_argument("-s", "--smooth", action="store_true")
parser.add_argument("-sf", "--smoothingfactor", action="store")
parser.add_argument("-avg", "--average", action="store_true")
parser.add_argument("-i2d", "--integral2d", action="store_true")
parser.add_argument("-un", "--usenegatives", action="store_true")
parser.add_argument("-otp", "--onetimeplot", action="store_true")
parser.add_argument("-pi", "--plotindex", action="store")

args = parser.parse_args()

frame_time = args.frametime
if frame_time is None:
    frame_time = 1000

smoothing_fac = args.smoothingfactor
if smoothing_fac is None:
    smoothing_fac = 2
smoothing_fac = int(smoothing_fac)

plot_i = args.plotindex
if plot_i is None:
    plot_i = -1
plot_i = int(plot_i)

################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

#################
## GET Surf EN ##
#################

suffix = ""
if args.average:
    suffix = "_avg"
elif args.integral2d:
    suffix = "_int2d"

if args.usenegatives:
    suffix += "_n"

df: pd.DataFrame = pd.read_csv(f"{sim_path}/surf_en{suffix}.csv", index_col=0)
df = df.apply(pd.to_numeric)

thetas_str = df.columns.to_numpy()
thetas_str = np.array(thetas_str)
thetas = thetas_str.astype(float)

################
## Calc Stiff ##
################

o_thetas_len = thetas.shape[0]

df_stiff = pd.DataFrame(columns=df.columns, index=df.index)
for i, row in df.iterrows():

    surf = row.to_numpy().copy()
    surf = np.hstack([surf, surf, surf])

    stiff = calc.calc_stiffness(surf, thetas)

    df_stiff.loc[i, :] = stiff[o_thetas_len : 2 * o_thetas_len]

###########
## FILL ##
###########


def fill(arr, div, add=False):

    o_arr = arr.copy()
    do_add_int = int(add)
    max_ = np.max(o_arr)

    for i in range(1, div):
        add_arr = do_add_int * i * max_
        arr = np.hstack([arr, o_arr + add_arr])

    return arr


if args.fill:
    thetas = fill(thetas, config["theta_div"], True)

################
## Plot setup ##
################

fig = plt.figure(figsize=(8, 5))
ax_surf = plt.subplot(121, projection="polar")
ax_stiff = plt.subplot(122, projection="polar")

ax_surf.set_aspect("equal")
ax_stiff.set_aspect("equal")

index = plot_i
if index < 0:
    index = df.shape[0] - np.abs(index)


def plot(frame):

    global smoothing_fac
    global args
    global thetas, config
    global ax_surf, ax_stiff
    global index, df, df_stiff

    if index == df.shape[0]:
        index = 0

    ax_stiff.cla()
    ax_surf.cla()

    surf = df.loc[index, :].to_numpy().copy()
    stiff = df_stiff.loc[index, :].to_numpy().copy()

    if args.smooth:

        o_stiff_len = stiff.shape[0]

        stiff = np.hstack([stiff, stiff, stiff])

        stiff = scipy.signal.savgol_filter(
            stiff, np.max([o_stiff_len // smoothing_fac, 4]), 3
        )

        stiff = stiff[o_stiff_len : 2 * o_stiff_len]

    if args.fill:
        surf = fill(surf, config["theta_div"])
        stiff = fill(stiff, config["theta_div"])

    ax_surf.plot(thetas, surf)
    ax_stiff.plot(thetas, stiff)

    ax_surf.set_ylim([np.min([0, np.min(surf)]), np.max(surf) + 0.5])
    ax_stiff.set_ylim([np.min([0, np.min(stiff)]), np.max(stiff) + 0.5])

    ax_surf.set_xticks([])
    ax_surf.set_yticks([])

    ax_stiff.set_xticks([])
    ax_stiff.set_yticks([])

    ax_surf.set_title(f"Surface Energy\nIteration: {index * config['writeEvery']}")
    ax_stiff.set_title("Stiffness")

    index += 1


if args.onetimeplot:

    plot("")

else:

    ani = FuncAnimation(plt.gcf(), plot, interval=frame_time)

plt.show()
