import argparse
import json
import sys

import numpy as np
import pandas as pd
from sim import calculations as calc
from sim import read_write as rw
from sim import utils

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="CalcGrainSurfEn", description="FFT Calc surf en for grain simulation"
)

parser.add_argument("sim_path")

args = parser.parse_args()

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

G = np.array(config["G"])

theta_count = config["theta_count"]
thetas = np.linsapce(0.0, 2.0 * np.pi, theta_count)

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)

eta_iter = rw.EtaIterator(
    sim_path, config["numPts"], config["numPts"], G.shape[0], float
)

iter_lines = eta_iter.count_lines()

theta_strs = [f"{theta:.4f}" for theta in thetas]
df = pd.DataFrame(columns=theta_strs, index=range(iter_lines))

#################
## CALCULATION ##
#################

for eta_iter_i, etas in enumerate(eta_iter):

    df.loc[eta_iter_i, :] = calc.calc_surf_en_2d(xm, ym, etas, thetas, config)

    perc = (eta_iter_i + 1) / iter_lines * 100
    sys.stdout.write(f"Progress: {perc:6.2f}% \r")
    sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()

##########
## Save ##
##########

df.to_csv(f"{sim_path}/surf_en_grain.csv")
