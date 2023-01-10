import argparse
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sim import calculations as calc
from sim import read_write as rw
from sim import utils

matplotlib.use("TkAgg")
sns.set_theme()
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="FFTPlotLineSurfEn", description="FFT Plot surf en for line simulation"
)

parser.add_argument("sim_path")
parser.add_argument("-pi", "--plotindex", action="store")
parser.add_argument("-f", "--fill", action="store_true")

args = parser.parse_args()

plot_i = args.plotindex
if plot_i is None:
    plot_i = -1

################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

##############
## GET VARS ##
##############

G = np.array(config["G"])
eta_path = f"{sim_path}/eta_files"
df = pd.read_csv(f"{sim_path}/surf_en.csv", index_col=0)

thetas_str = df.columns.to_numpy()
thetas_str = np.array(thetas_str)
thetas = thetas_str.astype(float)

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])

##########
## PLOT ##
##########

# plot and save

etas = rw.read_all_etas_at_line(
    f"{eta_path}/{thetas_str[plot_i]}",
    plot_i,
    config["numPts"],
    config["numPts"],
    G.shape[0],
    float,
)

eta_sum = np.zeros(etas[0].shape, dtype=complex)
for eta_i in range(G.shape[0]):
    eta_sum += etas[eta_i] * np.conj(etas[eta_i])
eta_sum = np.real(eta_sum).astype(float)

surf_en = df.iloc[plot_i, :].to_numpy().astype(float)
stiff = calc.calc_stiffness(surf_en, thetas)

if args.fill:

    o_thetas = thetas.copy()
    o_stiff = stiff.copy()
    o_surf_en = surf_en.copy()

    theta_max = thetas[-1]

    for i in range(1, config["theta_div"]):
        thetas = np.hstack([thetas, o_thetas + i * theta_max])

        stiff = np.hstack([stiff, o_stiff])
        surf_en = np.hstack([surf_en, o_surf_en])

xm, ym = np.meshgrid(x, x)

fig = plt.figure(figsize=(10, 5))

ax_main = plt.subplot(221)
ax_main.set_aspect("equal")
ax_stiff = plt.subplot(222, projection="polar")
ax_stiff.set_aspect("equal")
ax_surf_en = plt.subplot(224, projection="polar")
ax_surf_en.set_aspect("equal")
ax_text = plt.subplot(223)
ax_text.set_aspect("equal")

cont = ax_main.contourf(xm, ym, eta_sum, 100)
plt.colorbar(cont)

ax_stiff.plot(thetas, stiff)
ax_stiff.set_title("Stiffness")
ax_surf_en.plot(thetas, surf_en)
ax_surf_en.set_title("Surface Energy")

plt.show()
