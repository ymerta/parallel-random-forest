#!/bin/bash

# Derle
mpicc -o randomforest main_mpi.c load_data.c tree/tree.c

# Denenecek çekirdek sayıları
for np in 1 2 4 8
do
    echo "====================================="
    echo "Running with $np MPI process(es)"
    echo "====================================="
    mpirun -np $np ./randomforest
    echo ""
done