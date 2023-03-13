import sys
import time
import datetime
import argparse

from sim import fft_sim

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

args = parser.parse_args()

###############
## SETUP SIM ##
###############

thread_list = fft_sim.initialize_sim_threads(
    args.sim_path,
    args.calcdb0,
    not args.skipetacalc,
    args.calcn0,
    args.threadcount,
    args.continuesim,
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
