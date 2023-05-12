#!/bin/sh

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
chmod +x ./DFL_simulator_mt
./DFL_simulator_mt