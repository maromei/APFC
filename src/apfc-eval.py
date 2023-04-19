import os
import argparse

import numpy as np
import pandas as pd

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

parser.add_argument(
    "-pes",
    "--phase-eq-strategy",
    action="store",
    type=str,
    help=(
        "Strategy to determine how the equilibrium values are determined."
        " It will override the strategy found in the config."
        " Possible values are: 'minmax', 'maxmin' and 'firstlast'."
    ),
)

args = parser.parse_args()

####################
## PREP VARIABLES ##
####################


def norm(arr):
    ret = arr
    # ret = arr - np.min(arr)
    return ret / np.max(ret)


def run_evaluation(sim_path: str, phase_eq_func):

    config = utils.get_config(sim_path)

    sim_path = utils.make_path_arg_absolute(sim_path)
    out_dir = f"{sim_path}/evaluate"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    eta_count = len(config["G"])
    include_n0 = config["simType"] == "n0"
    is_1d = config["numPtsY"] <= 1
    line_count = rw.count_lines(f"{sim_path}/eta_files/0.0000/out_0.txt")

    x = np.linspace(-config["xlim"], config["xlim"], config["numPtsX"])

    ############################
    ## BUILD EMPTY DATAFRAMES ##
    ############################

    thetas = utils.get_thetas(config)
    thetas_str = [f"{theta:.4f}" for theta in thetas]

    df_surf_en = pd.DataFrame(columns=thetas_str, index=range(line_count))

    eval_columns = [
        "eqEtaSolid",
        "eqEtaLiquid",
        "eqN0Solid",
        "eqN0Liquid",
        "interfaceWidth",
        "radius",
    ]
    eval_index = pd.MultiIndex.from_product(
        [[i for i in range(line_count)], eval_columns]
    )
    df_eval = pd.DataFrame(columns=thetas_str, index=eval_index)

    #####################
    ## THETA ITERATION ##
    #####################

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

        theta_col_name = thetas_str[theta_i]

        #####################
        ## ENTRY ITERATION ##
        #####################

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

            surf_en = observables.calc_surf_en_1d(
                etas, n0, config, theta, phase_eq_func=phase_eq_func
            )
            df_surf_en.iloc[entry_i, theta_i] = surf_en

            ######################
            ## Eval Observables ##
            ######################

            eta_sum = np.zeros(etas[0].shape)
            for eta_i in range(eta_count):
                eta_sum += etas[eta_i] ** 2

            x_pos, eta_sum = utils.get_positive_range(x, eta_sum)

            eta_s, eta_l = (np.max(eta_sum), np.min(eta_sum))  # phase_eq_func(eta_sum)
            radius, intWidth = observables.fit_to_tanhmin(x_pos, eta_sum, False)

            n0_s = n0
            n0_l = n0

            if include_n0:
                n0_s, n0_l = phase_eq_func(n0)

            df_eval.loc[(entry_i, "eqEtaSolid"), theta_col_name] = eta_s
            df_eval.loc[(entry_i, "eqEtaLiquid"), theta_col_name] = eta_l
            df_eval.loc[(entry_i, "eqN0Solid"), theta_col_name] = n0_s
            df_eval.loc[(entry_i, "eqN0Liquid"), theta_col_name] = n0_l
            df_eval.loc[(entry_i, "interfaceWidth"), theta_col_name] = intWidth
            df_eval.loc[(entry_i, "radius"), theta_col_name] = radius

    ######################
    ## Widen DataFrames ##
    ######################

    thetas = utils.fill(thetas, config["thetaDiv"], True)
    thetas_str = [f"{theta:.4f}" for theta in thetas]

    df_surf_en = utils.fill_df(df_surf_en, thetas_str, config["thetaDiv"])
    df_eval = utils.fill_df(df_eval, thetas_str, config["thetaDiv"])

    #######################
    ## Stiffness and Fits##
    #######################

    df_stiff = pd.DataFrame(columns=thetas_str, index=range(line_count))
    df_fits = pd.DataFrame(columns=["eps", "gamma0"], index=range(line_count))
    df_fits_norm = pd.DataFrame(columns=["eps", "gamma0"], index=range(line_count))

    for i in range(df_surf_en.shape[0]):
        df_stiff.iloc[i, :] = observables.calc_stiffness(df_surf_en.iloc[i, :], thetas)

        surf_en_norm = norm(df_surf_en.iloc[i, :].to_numpy())

        try:
            df_fits.iloc[i, :] = triangular.fit_surf_en(thetas, df_surf_en.iloc[i, :])
            df_fits_norm.iloc[i, :] = triangular.fit_surf_en(thetas, surf_en_norm)
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
    df_fits_norm.to_csv(f"{out_dir}/fits_norm.csv")
    df_eval.to_csv(f"{out_dir}/eval.csv")


#################
## Handle Vary ##
#################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config = utils.get_config(sim_path)

phase_eq_strategy = config.get("phase_eq_strategy", "firstlast")
if args.phase_eq_strategy is not None:
    phase_eq_strategy = args.phase_eq_strategy

phase_eq_func = observables.get_phase_eq_values
if phase_eq_strategy == "firstlast":
    phase_eq_func = observables.get_phase_eq_values
elif phase_eq_strategy == "minmax":
    phase_eq_func = lambda x: (np.min(x), np.max(x))
elif phase_eq_strategy == "maxmin":
    phase_eq_func = lambda x: (np.max(x), np.min(x))
else:
    print(
        (
            "Invalid phase eq strategy found. "
            "See '--help' for infos. "
            "Defaulting to 'firstlast'."
        )
    )


if config.get("vary", False):

    vary_path = f"{sim_path}/{config['varyParam']}"
    vary_values = utils.read_vary_vals_from_dir(vary_path, config["varyParam"])

    for vary_i, vary_val in enumerate(vary_values):
        print(
            f"Running Evaluation for "
            f"{config['varyParam']}={utils.get_vary_val_dir_name(vary_val)} "
            f"({vary_i + 1}/{vary_values.shape[0]})."
        )
        new_sim_path = f"{vary_path}/{utils.get_vary_val_dir_name(vary_val)}"
        run_evaluation(new_sim_path, phase_eq_func)

else:

    run_evaluation(sim_path, phase_eq_func)
