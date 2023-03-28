import os
import shutil
import sys
import time
import datetime
import argparse
import json

import numpy as np

from sim import fft_sim, parameter_sets
from manage import utils

####################
## SETUP ARGPARSE ##
####################

parser = argparse.ArgumentParser(prog="FFTSim", description="FFT Simluation")

parser.add_argument(
    "sim_path",
    help=(
        "The path where the config file lies,"
        "and where the output should be generated."
    ),
)

parser.add_argument(
    "--calcdb0", action="store_true", help="Flag to calculate the equilibrium dB0."
)

parser.add_argument(
    "--skipetacalc",
    action="store_true",
    help=(
        "Whether the calculation of the initial amplitudes should be skipped."
        "The 'initEta' key will then be used for initialization."
    ),
)

parser.add_argument(
    "--calcn0",
    action="store_true",
    help=(
        "Whether the equilibrium average density"
        " for the given amplitude should be calculated."
    ),
)

parser.add_argument(
    "-tc",
    "--threadcount",
    action="store",
    type=int,
    default=1,
    help="How many threads should be created.",
)

parser.add_argument(
    "-con",
    "--continuesim",
    action="store_true",
    help="Should this simulation continue by reading the last out-files.",
)

parser.add_argument(
    "-aps",
    "--applyparamset",
    action="store",
    type=int,
    help="Applies values from the parameterset with the given index.",
)

args = parser.parse_args()

######################
## RUN SIM FUNCTION ##
######################


def run_sim(
    sim_path: str,
    calcdb0: bool,
    skipetacalc: bool,
    calcn0: bool,
    threadcount: int,
    continuesim: bool,
    applyparamset: bool,
):

    thread_list = fft_sim.initialize_sim_threads(
        sim_path,
        calcdb0,
        not skipetacalc,
        calcn0,
        threadcount,
        continuesim,
        applyparamset,
    )

    start_time = time.time()
    curr_date_time = datetime.datetime.now()
    print(f"Starting Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")

    #############
    ## RUN SIM ##
    #############

    for i in range(len(thread_list)):
        thread_list[i].start()

    while fft_sim.are_processes_running(thread_list):
        fft_sim.print_progress_string()
        time.sleep(5)

    sys.stdout.write(f"\n")
    sys.stdout.flush()

    for i in range(len(thread_list)):
        thread_list[i].join()

    ################
    ## FINISH SIM ##
    ################

    time_diff = int(time.time() - start_time)
    hours = time_diff // (60.0 * 60)
    time_diff -= hours * 60 * 60
    minutes = time_diff // 60
    time_diff -= minutes * 60
    seconds = time_diff

    print(f"Time: {int(hours)}:{int(minutes)}:{int(seconds)}")

    curr_date_time = datetime.datetime.now()
    print(f"End Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")


################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config = utils.get_config(sim_path)

#################
## HANDLE VARY ##
#################

if config.get("vary", False):

    ### Get relevant values ###

    vary_values = np.linspace(
        config["varyStart"], config["varyEnd"], config["varyAmount"]
    )

    vary_path = f"{sim_path}/{config['varyParam']}"

    ### Remove Dir incase it exists ###

    if os.path.exists(vary_path):
        shutil.rmtree(vary_path)
    os.makedirs(vary_path)

    ### Time ###

    start_time = time.time()
    curr_date_time = datetime.datetime.now()
    print(
        "########\n"
        f"Starting Time with varying {config['varyParam']}: "
        f"{curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "########\n"
    )

    for vary_i, vary_val in enumerate(vary_values):

        ### build new config ###

        new_sim_path = f"{vary_path}/{utils.get_vary_val_dir_name(vary_val)}"
        new_config_path = f"{new_sim_path}/config.json"

        os.makedirs(new_sim_path)

        new_config = config.copy()
        # just to be save in avoiding recursive behaviour
        config["vary"] = False
        config["simPath"] = new_sim_path
        config[config["varyParam"]] = vary_val

        with open(new_config_path, "w+") as f:
            json.dump(config, f)

        ### Run Sim ###

        print(
            f"Running "
            f"{config['varyParam']}={utils.get_vary_val_dir_name(vary_val)} "
            f"({vary_i + 1}/{vary_values.shape[0]})."
        )

        run_sim(
            new_sim_path,
            args.calcdb0,
            args.skipetacalc,
            args.calcn0,
            args.threadcount,
            args.continuesim,
            args.applyparamset,
        )

        print("")  # Just one empty line for better terminal readability.

    ################
    ## FINISH SIM ##
    ################

    time_diff = int(time.time() - start_time)
    hours = time_diff // (60.0 * 60)
    time_diff -= hours * 60 * 60
    minutes = time_diff // 60
    time_diff -= minutes * 60
    seconds = time_diff

    print(f"Time: {int(hours)}:{int(minutes)}:{int(seconds)}")

    curr_date_time = datetime.datetime.now()
    print(f"End Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(
        "\n########\n"
        f"Total time: {int(hours)}:{int(minutes)}:{int(seconds)}\n"
        f"End Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "########"
    )

else:

    run_sim(
        args.sim_path,
        args.calcdb0,
        args.skipetacalc,
        args.calcn0,
        args.threadcount,
        args.continuesim,
        args.applyparamset,
    )
