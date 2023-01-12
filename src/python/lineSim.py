import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from sim import eta_builder, rfft_sim, utils

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="FFTLineSSim", description="FFT Simluation of center line sim"
)

parser.add_argument("sim_path")
parser.add_argument("-cdb0", "--calcdb0", action="store_true")
parser.add_argument("-cie", "--calciniteta", action="store_true")
parser.add_argument("-cie2m", "--calciniteta2m", action="store_true")
parser.add_argument("-cie2p", "--calciniteta2p", action="store_true")
parser.add_argument("-csimp", "--calcsimplevariables", action="store_true")
parser.add_argument("-con", "--continuesim", action="store_true")
parser.add_argument("-tc", "--threadcount", action="store")
parser.add_argument("-ull", "--useleftline", action="store_true")
parser.add_argument("-ug", "--usegrain", action="store_true")

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

n0 = config["n0"]
t = config["t"]
v = config["v"]
Bx = config["Bx"]
dB0 = config["dB0"]

if args.calcdb0:

    db0_fac = config.get("dB0_fac", 1.0)

    dB0 = 8.0 * t**2 / (135.0 * v) * db0_fac
    config["dB0"] = dB0

if args.calcsimplevariables:

    config["A"] = Bx
    config["B"] = dB0 - 2.0 * t * n0 + 3.0 * v * n0**2
    config["C"] = -t - 3.0 * n0
    config["D"] = v

if args.calciniteta:
    config["initEta"] = 4.0 * t / (45.0 * v)

if args.calciniteta2m:
    config["initEta"] = (t - np.sqrt(t**2 - 15.0 * v * dB0)) / 15.0 * v

if args.calciniteta2p:
    config["initEta"] = (t + np.sqrt(t**2 - 15.0 * v * dB0)) / 15.0 * v

config["sim_path"] = sim_path

if args.useleftline:
    config["initType"] = "leftLine"
elif args.usegrain:
    config["initType"] = "grain"
else:
    config["initType"] = "centerLine"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

#####################
## SIMULTAION VARS ##
#####################

thetas = np.linspace(
    0, 2.0 * np.pi / config["theta_div"], config["theta_count"], endpoint=True
)

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

    for theta_i, theta in enumerate(thetas):

        ### create direcotry if not exist ###

        theta_path = f"{eta_path}/{theta:.4f}"
        if not os.path.exists(theta_path):
            os.makedirs(theta_path)

        ### rotate G in config ###

        # fmt: off
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # fmt: on

        G = np.array(config["G"])
        for eta_i in range(G.shape[0]):
            G[eta_i] = rot.dot(G[eta_i])
        config["G"] = G.tolist()

        if continue_sim:
            config["sim_path"] = theta_path
            sim = rfft_sim.FFTSim(config, eta_builder.load_from_file)
            ignore_first_write = True
        else:
            if args.useleftline:
                sim = rfft_sim.FFTSim(config, eta_builder.left_line)
            elif args.usegrain:
                sim = rfft_sim.FFTSim(config, eta_builder.single_grain)
            else:
                sim = rfft_sim.FFTSim(config, eta_builder.center_line)
            sim.reset_out_files(theta_path)
            ignore_first_write = False

        ### run sim ###

        for i in range(step_count + 1):  # +1 to get first and last write

            sim.run_one_step()

            should_write = i % write_every_i == 0
            should_write = should_write and not (i == 0 and ignore_first_write)

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
