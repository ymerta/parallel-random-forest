#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "tree/tree.h"

#define MAX_ROWS 10000
#define MAX_FEATURES 200
#define FEATURE_SUBSET 30
#define N_TOTAL_TREES 100
#define TEST_START_INDEX 8000
#define N_TEST 300

extern float features[MAX_ROWS][MAX_FEATURES];
extern int labels[];
extern void load_csv(const char *filename, int *num_rows);

int main(int argc, char **argv)
{
    int rank, size, num_rows;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    int N_TREES_PER_RANK = N_TOTAL_TREES / size;

    if (rank == 0)
    {
        printf("Rank 0 veriyi yüklüyor...\n");
        load_csv("dataset/train.csv", &num_rows);
    }
    // Rank 0 işlemcisi train.csv dosyasını yüklüyor. Diğer işlemciler bu veriyi doğrudan dosyadan okumuyor.
    // Bunun yerine Rank 0, veriyi okuduktan sonra MPI_Bcast ile tüm işlemcilere dağıtıyor.
    MPI_Bcast(&num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(features, MAX_ROWS * MAX_FEATURES, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(labels, MAX_ROWS, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Rank %d: num_rows = %d, training %d trees\n", rank, num_rows, N_TREES_PER_RANK);

    srand((unsigned int)time(NULL) ^ (rank << 16));
    DecisionStump *forest = malloc(sizeof(DecisionStump) * N_TREES_PER_RANK);

    for (int t = 0; t < N_TREES_PER_RANK; t++)
    {
        int selected_features[FEATURE_SUBSET];
        for (int i = 0; i < FEATURE_SUBSET; i++)
        {
            selected_features[i] = rand() % MAX_FEATURES;
        }

        forest[t] = train_random_stump(features, labels, num_rows, selected_features, FEATURE_SUBSET);

        printf("Rank %d → Tree %d: Feature = %d, Threshold = %.3f, Left = %d, Right = %d\n",
               rank, t, forest[t].feature_index, forest[t].threshold,
               forest[t].prediction_left, forest[t].prediction_right);
    }

    int correct = 0;

    for (int test_idx = 0; test_idx < N_TEST; test_idx++)
    {
        float *test_sample = features[TEST_START_INDEX + test_idx];
        int true_label = labels[TEST_START_INDEX + test_idx];

        int *local_predictions = malloc(sizeof(int) * N_TREES_PER_RANK);
        for (int i = 0; i < N_TREES_PER_RANK; i++)
        {
            local_predictions[i] = predict_with_stump(forest[i], test_sample);
        }

        int *all_predictions = NULL;
        if (rank == 0)
            all_predictions = malloc(sizeof(int) * N_TOTAL_TREES);
        // Her işlemci kendi DecisionStump modeliyle test örneğine tahmin yapar (local_predictions array’i).
        // Bu tahminler MPI_Gather kullanılarak rank 0’da toplanır.
        MPI_Gather(local_predictions, N_TREES_PER_RANK, MPI_INT,
                   all_predictions, N_TREES_PER_RANK, MPI_INT,
                   0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            int zero = 0, one = 0;
            for (int i = 0; i < N_TOTAL_TREES; i++)
            {
                if (all_predictions[i] == 0)
                    zero++;
                else
                    one++;
            }

            int final_prediction = (one > zero) ? 1 : 0;
            if (final_prediction == true_label)
                correct++;

            free(all_predictions);
        }

        free(local_predictions);
    }

    if (rank == 0)
    {
        float accuracy = (float)correct / N_TEST;
        printf("\n✅ Accuracy over %d samples: %.2f%% (%d/%d correct)\n", N_TEST, accuracy * 100, correct, N_TEST);
    }

    free(forest);

    double end_time = MPI_Wtime();
    if (rank == 0)
    {
        printf("⏱️ Elapsed Time: %.6f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}