import argparse
import json
import sys
import datetime
import time

import numpy as np
from sim import eta_builder, rfft_sim, fft_n0, utils

#####################
## SETUP ARGUMENTS ##
#####################

parser = argparse.ArgumentParser(
    prog="FFTGrainSim", description="FFT Simluation of simple grain"
)

parser.add_argument("sim_path")
parser.add_argument("-cdb0", "--calcdb0", action="store_true")
parser.add_argument("-con", "--continuesim", action="store_true")
parser.add_argument("-n0", "--n0", action="store_true")

args = parser.parse_args()

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

config["simType"] = "grain"

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

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

####################
## RUN SIMULTAION ##
####################

start_time = time.time()

curr_date_time = datetime.datetime.now()
print(f"Starting Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")

if args.n0:
    SimClass = fft_n0.FFTSim
else:
    SimClass = rfft_sim.FFTSim

if args.continuesim:
    sim = SimClass(config, eta_builder.load_from_file)
    ignore_first_write = True
else:
    sim = SimClass(config, eta_builder.single_grain)
    sim.reset_out_files(sim_path)
    ignore_first_write = False

step_count: int = config["numT"]
write_every_i: int = config["writeEvery"]

if not ignore_first_write:
    sim.write(sim_path)

for i in range(step_count + 1):  # +1 to get first and last write

    sim.run_one_step()

    should_write = i % write_every_i == 0 and i != 0
    if not should_write:
        continue

    sim.write(sim_path)

    perc = i / step_count * 100
    sys.stdout.write(f"Progress: {perc:5.1f}%\r")
    sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()

time_diff = int(time.time() - start_time)
hours = time_diff // (60.0 * 60)
time_diff -= hours * 60 * 60
minutes = time_diff // 60
time_diff -= minutes * 60
seconds = time_diff

print(f"Time: {int(hours)}:{int(minutes)}:{int(seconds)}")

curr_date_time = datetime.datetime.now()
print(f"End Time: {curr_date_time.strftime('%Y-%m-%d %H:%M:%S')}")
