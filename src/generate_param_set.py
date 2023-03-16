import os
import argparse
import json

from manage import utils
from sim import parameter_sets

####################
## SETUP ARGPARSE ##
####################

parser = argparse.ArgumentParser(
    prog="GenerateParamSet", description="Generates Parameter sets"
)

parser.add_argument(
    "sim_path",
    help=(
        "The path where the config file lies,"
        "and where the output should be generated."
    ),
)

parser.add_argument(
    "-psi",
    "--paramsetid",
    action="store",
    type=int,
    default=0,
    help="Applies values from the parameterset with the given index.",
)

parser.add_argument(
    "-n0",
    "--n0",
    action="store_true",
    help="Whether it should initialize with n0 simType.",
)

args = parser.parse_args()

###############
## SETUP DIR ##
###############

sim_path = utils.make_path_arg_absolute(args.sim_path)
config_path = f"{sim_path}/config.json"

if not os.path.exists(sim_path):
    os.makedirs(sim_path)

config = parameter_sets.DEFAULT_CONFIG.copy()

######################
## APPLY PARAMETERS ##
######################

param_i = args.paramsetid

for key, value in parameter_sets.PARAM_SETS[param_i].items():
    config[key] = value

if args.n0:
    config["simType"] = "n0"

config["simPath"] = sim_path

#################
## SAVE CONFIG ##
#################

with open(config_path, "w+") as f:
    json.dump(config, f, indent=4)
