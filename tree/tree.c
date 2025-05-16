#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "tree.h"

int majority_class(int* labels, int* indices, int count) {
    int zero = 0, one = 0;
    for (int i = 0; i < count; i++) {
        if (labels[indices[i]] == 0) zero++;
        else one++;
    }
    return (one > zero) ? 1 : 0;
}

int random_int(int max) {
    return rand() % max;
}

int predict_with_stump(DecisionStump stump, float* sample) {
    if (sample[stump.feature_index] < stump.threshold)
        return stump.prediction_left;
    else
        return stump.prediction_right;
}

DecisionStump train_random_stump(float features[][200], int labels[], int num_rows, int* selected_features, int num_features) {
    DecisionStump best = {0, 0.0f, 0, 0};
    float best_accuracy = 0.0;

    for (int f = 0; f < num_features; f++) {
        int feature = selected_features[f];
        for (int r = 0; r < num_rows; r += num_rows / 10) {
            float threshold = features[r][feature];

            int left_count = 0, right_count = 0;
            int left_indices[num_rows], right_indices[num_rows];

            for (int i = 0; i < num_rows; i++) {
                if (features[i][feature] < threshold)
                    left_indices[left_count++] = i;
                else
                    right_indices[right_count++] = i;
            }

            if (left_count == 0 || right_count == 0)
                continue;

            int left_pred = majority_class(labels, left_indices, left_count);
            int right_pred = majority_class(labels, right_indices, right_count);

            int correct = 0;
            for (int i = 0; i < num_rows; i++) {
                int pred = (features[i][feature] < threshold) ? left_pred : right_pred;
                if (pred == labels[i]) correct++;
            }

            float acc = (float)correct / num_rows;
            if (acc > best_accuracy) {
                best_accuracy = acc;
                best.feature_index = feature;
                best.threshold = threshold;
                best.prediction_left = left_pred;
                best.prediction_right = right_pred;
            }
        }
    }

    return best;
}