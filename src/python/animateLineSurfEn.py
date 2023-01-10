import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

args = parser.parse_args()

frame_time = args.frametime
if frame_time is None:
    frame_time = 1000

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

df: pd.DataFrame = pd.read_csv(f"{sim_path}/surf_en.csv", index_col=0)
df = df.apply(pd.to_numeric)

thetas_str = df.columns.to_numpy()
thetas_str = np.array(thetas_str)
thetas = thetas_str.astype(float)

################
## Calc Stiff ##
################

df_stiff = pd.DataFrame(columns=df.columns, index=df.index)
for i, row in df.iterrows():
    df_stiff.loc[i, :] = calc.calc_stiffness(row.to_numpy().copy(), thetas)

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

index = 0


def plot(frame):

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

    if args.fill:
        surf = fill(surf, config["theta_div"])
        stiff = fill(stiff, config["theta_div"])

    ax_surf.scatter(thetas, surf)
    ax_stiff.scatter(thetas, stiff)

    ax_surf.set_xticks([])
    ax_surf.set_yticks([])

    ax_stiff.set_xticks([])
    ax_stiff.set_yticks([])

    ax_surf.set_title(f"Surface Energy\nIteration: {index * config['writeEvery']}")
    ax_stiff.set_title("Stiffness")

    index += 1


ani = FuncAnimation(plt.gcf(), plot, interval=frame_time)

plt.show()
