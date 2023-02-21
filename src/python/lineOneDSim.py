import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import datetime

import numpy as np
from sim import eta_builder, fft_sim_1d, utils, fft_n0

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="FFTLineSSim", description="FFT Simluation of center line sim"
)

parser.add_argument("sim_path")
parser.add_argument("-cdb0", "--calcdb0", action="store_true")
parser.add_argument("-con", "--continuesim", action="store_true")
parser.add_argument("-tc", "--threadcount", action="store")
parser.add_argument("-n0", "--n0", action="store_true")

args = parser.parse_args()

thread_count = args.threadcount
if thread_count is None:
    thread_count = 1
thread_count = int(thread_count)

################
## GET CONFIG ##
################

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

with open(config_path, "r") as f:
    config = json.load(f)

######################
## HANDLE ARGUMENTS ##
######################

config["simType"] = "line1d"

n0 = config["n0"]
t = config["t"]
v = config["v"]
Bx = config["Bx"]
dB0 = config["dB0"]

if args.calcdb0:

    db0_fac = config.get("dB0_fac", 1.0)

    dB0 = 8.0 * t**2 / (135.0 * v) * db0_fac
    config["dB0"] = dB0

eta_builder.init_config(config)
eta_builder.init_eta_height(config)

config["sim_path"] = sim_path

config["initType"] = "centerLine"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

#####################
## SIMULTAION VARS ##
#####################

thetas = utils.get_thetas(config)

eta_path = f"{sim_path}/eta_files"
if not os.path.exists(eta_path):
    os.makedirs(eta_path)

#########################
## SIMULATION FUNCTION ##
#########################


def theta_thread(thetas, config, eta_path, continue_sim, index, args):

    step_count: int = config["numT"]
    write_every_i: int = config["writeEvery"]

    total_steps = thetas.shape[0] * step_count

    G_original = np.array(config["G"])
    config_original = config.copy()

    for theta_i, theta in enumerate(thetas):

        ### create direcotry if not exist ###

        theta_path = f"{eta_path}/{theta:.4f}"
        if not os.path.exists(theta_path):
            os.makedirs(theta_path)

        config = config_original.copy()
        config["sim_path"] = theta_path

        ### rotate G in config ###

        # fmt: off
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # fmt: on

        G = G_original.copy()
        for eta_i in range(G.shape[0]):
            G[eta_i] = rot.dot(G[eta_i])
        config["G"] = G.tolist()

        if args.n0:
            SimClass = fft_n0.FFTSim
        else:
            SimClass = fft_sim_1d.FFTSim

        sim = SimClass(config, continue_sim)
        ignore_first_write = continue_sim

        if not continue_sim:
            sim.reset_out_files(theta_path)

        ### run sim ###

        for _ in range(config.get("eqSteps", 0)):
            sim.run_one_step()

        if not ignore_first_write:
            sim.write(theta_path)

        for i in range(step_count + 1):  # +1 to get first and last write

            sim.run_one_step()

            should_write = i % write_every_i == 0 and i != 0
            if not should_write:
                continue

            sim.write(theta_path)

            curr_i = theta_i * (step_count + 1) + i + 1
            perc = curr_i / total_steps * 100
            progr_lst[index] = perc


#####################
## MULTI THREADING ##
#####################

# progress list
progr_lst = [0.0 for _ in range(thread_count)]
mp_manager = mp.Manager()
progr_lst = mp_manager.list(progr_lst)

# generate theta lists

theta_lst = [[] for _ in range(thread_count)]

for theta_i, theta in enumerate(thetas):
    theta_lst[theta_i % thread_count].append(theta)

# generate thread_list

thread_lst = []
for i in range(thread_count):
    thread_lst.append(
        mp.Process(
            target=theta_thread,
            args=(
                np.array(theta_lst[i]),
                config.copy(),
                eta_path,
                args.continuesim,
                i,
                args,
            ),
        )
    )

start_time = time.time()

curr_date_time = datetime.datetime.now()
print(f"Starting Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")

for i in range(thread_count):
    thread_lst[i].start()


def are_mp_running(thread_lst):
    for i in range(len(thread_lst)):
        if thread_lst[i].is_alive():
            return True

    return False


while are_mp_running(thread_lst):

    prog_strs = [f"{n:5.1f}%" for n in progr_lst]
    prog_strs = " ".join(prog_strs)
    sys.stdout.write(f"{prog_strs}\r")
    sys.stdout.flush()

    time.sleep(5)

sys.stdout.write(f"\n")
sys.stdout.flush()

for i in range(thread_count):
    thread_lst[i].join()

time_diff = int(time.time() - start_time)
hours = time_diff // (60.0 * 60)
time_diff -= hours * 60 * 60
minutes = time_diff // 60
time_diff -= minutes * 60
seconds = time_diff

print(f"Time: {int(hours)}:{int(minutes)}:{int(seconds)}")


curr_date_time = datetime.datetime.now()
print(f"End Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")
