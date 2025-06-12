#!/bin/bash

# Derle
mpicc -o randomforest main_mpi.c load_data.c tree/tree.c

# Sonuç dosyasını temizle
echo "Random Forest MPI Test Results" > results_mpi.txt

# Denenecek çekirdek sayıları
for np in 16 32 64 128
do
    echo "=====================================" | tee -a results_mpi.txt
    echo "Running with $np MPI process(es)" | tee -a results_mpi.txt
    echo "=====================================" | tee -a results_mpi.txt

    mpirun -np $np ./randomforest | tee -a results_mpi.txt
    echo "" | tee -a results_mpi.txt
done