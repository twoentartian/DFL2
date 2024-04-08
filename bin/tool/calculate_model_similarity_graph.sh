#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

chomd +x ./calculate_model_fusion_accuracy
./calculate_model_fusion_accuracy "$@"