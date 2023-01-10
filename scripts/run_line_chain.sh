#!/bin/bash

clear

echo -e "Using Directory: $1\n"

echo "Running Line Simulation..."
venv/bin/python src/python/lineSim.py $1 -cie -csim

echo -e "\nRunning Surface Energy Calculation..."
venv/bin/python src/python/calcLineSurfEn.py $1

echo -e "\nPlotting..."
venv/bin/python src/python/plotLineSim.py $1 --fill
