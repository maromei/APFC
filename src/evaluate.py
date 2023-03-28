import os
import argparse

import numpy as np
import pandas as pd
import scipy

from manage import utils
from manage import read_write as rw

from calculations import observables, triangular

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="Evaluate", description="Evaluates one simulation"
)

parser.add_argument(
    "sim_path",
    help=(
        "The path where the config file lies,"
        "and where the output should be generated."
    ),
)

args = parser.parse_args()

####################
## PREP VARIABLES ##
####################

config = utils.get_config(args.sim_path)

sim_path = utils.make_path_arg_absolute(args.sim_path)
out_dir = f"{sim_path}/evaluate"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

eta_count = len(config["G"])
include_n0 = config["simType"] == "n0"
is_1d = config["numPtsY"] <= 1
line_count = rw.count_lines(f"{sim_path}/eta_files/0.0000/out_0.txt")

x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])

###########
## EVALS ##
###########

thetas = utils.get_thetas(config)
thetas_str = [f"{theta:.4f}" for theta in thetas]

df_surf_en = pd.DataFrame(columns=thetas_str, index=range(line_count))

eval_columns = ["eqEtaSolid", "eqEtaLiquid", "eqN0Solid", "eqN0Liquid"]
eval_all = np.zeros((line_count, thetas.shape[0], 6))

for theta_i, theta in enumerate(thetas):

    theta_dir = f"{sim_path}/eta_files/{theta:.4f}"

    eta_it = rw.EtaIterator(
        theta_dir,
        config["numPtsX"],
        config["numPtsY"],
        eta_count,
        float,
        config["simType"] == "n0",
    )

    for entry_i, entry in enumerate(eta_it):

        if include_n0:
            etas, n0 = entry
        else:
            etas = entry
            n0 = config["n0"]

        # if it is 2d then we can only evaluate the center line
        if not is_1d:
            center_index = etas.shape[2] // 2
            etas = etas[:, center_index, :]

        if not is_1d and include_n0:
            center_index = etas.shape[2] // 2
            n0 = n0[center_index, :]

        ####################
        ## Surface Energy ##
        ####################

        surf_en = observables.calc_surf_en_1d(etas, n0, config, theta)
        df_surf_en.iloc[entry_i, theta_i] = surf_en

##########################
## Widen Surface Energy ##
##########################

thetas = utils.fill(thetas, config["thetaDiv"], True)
thetas_str = [f"{theta:.4f}" for theta in thetas]

df_surf_en = utils.fill_df(df_surf_en, thetas_str, config["thetaDiv"])

#######################
## Stiffness and Fits##
#######################

df_stiff = pd.DataFrame(columns=thetas_str, index=range(line_count))
df_fits = pd.DataFrame(columns=["eps", "gamma0"], index=range(line_count))

for i in range(df_surf_en.shape[0]):
    df_stiff.iloc[i, :] = observables.calc_stiffness(df_surf_en.iloc[i, :], thetas)

    try:
        df_fits.iloc[i, :] = triangular.fit_surf_en(thetas, df_surf_en.iloc[i, :])
    except Exception as e:
        print(
            f"Could not calculate the fits for index {i}/{line_count} because of the Error:"
        )
        print(e)
        print("Skipping...")

##########
## Save ##
##########

df_surf_en.to_csv(f"{out_dir}/surf_en.csv")
df_stiff.to_csv(f"{out_dir}/stiff.csv")
df_fits.to_csv(f"{out_dir}/fits.csv")
