#!/bin/sh

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
chmod +x ./DFL_simulator_mt
nice -n 1 ./DFL_simulator_mt