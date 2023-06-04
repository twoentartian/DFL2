#!/bin/sh

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1
chmod +x ./DFL_simulator_opti
./DFL_simulator_opti