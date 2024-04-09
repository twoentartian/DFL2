#!/bin/bash
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

chmod +x ./calculate_model_similarity_graph
./calculate_model_similarity_graph "$@"