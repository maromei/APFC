import argparse
import json
import pathlib
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
    prog="FFTCalcLineSurfEn", description="FFT Calc surf en for line simulation"
)

parser.add_argument("sim_path")
parser.add_argument("-avg", "--average", action="store_true")
parser.add_argument("-i2d", "--integral2d", action="store_true")
parser.add_argument("-un", "--usenegatives", action="store_true")

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

eta_path = f"{sim_path}/eta_files"
thetas = pathlib.Path(eta_path)
thetas = thetas.iterdir()
thetas_str = [str(p).split("/")[-1] for p in thetas]
thetas_str.sort()

thetas = np.array(thetas_str, dtype=float)

df_dic = {th_str: [] for th_str in thetas_str}

G = np.array(config["G"])

x = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
xm, ym = np.meshgrid(x, x)

#################
## CALCULATION ##
#################

for theta_i, theta in enumerate(thetas):

    ### iteration ###

    eta_iter = rw.EtaIterator(
        f"{eta_path}/{thetas_str[theta_i]}",
        config["numPts"],
        config["numPts"],
        G.shape[0],
        float,
    )

    iter_lines = eta_iter.count_lines()

    for eta_iter_i, etas in enumerate(eta_iter):

        surf_en = calc.calc_line_surf_en(
            xm,
            ym,
            config,
            etas,
            theta,
            rot_g=True,
            average=args.average,
            integ2d=args.integral2d,
            use_pos=not args.usenegatives,
        )

        df_dic[thetas_str[theta_i]].append(surf_en)

        ### Progress bar ###

        curr_comp = theta_i * iter_lines + eta_iter_i + 1
        perc = curr_comp / (thetas.shape[0] * iter_lines) * 100

        progress_str = (
            f"Working on Theta {theta:.4f} [{theta_i+1}/{thetas.shape[0]}]"
            f" Overall Progress: {perc:.4f}%\r"
        )

        sys.stdout.write(progress_str)
        sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()

# create df with proper index

df = pd.DataFrame(df_dic)

suffix = ""
if args.average:
    suffix = "_avg"
elif args.integral2d:
    suffix = "_int2d"

if args.usenegatives:
    suffix += "_n"

df.to_csv(f"{sim_path}/surf_en{suffix}.csv")
