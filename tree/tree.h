#ifndef TREE_H
#define TREE_H

typedef struct {
    int feature_index;   // Hangi özelliğe göre böldü
    float threshold;     // Eşik değer
    int prediction_left;  // eşikten küçük/veri için tahmin
    int prediction_right; // eşikten büyük/veri için tahmin
    
} DecisionStump;


int predict_with_stump(DecisionStump stump, float* sample);

DecisionStump train_random_stump(float features[][200], int labels[], int num_rows, int* selected_features, int num_features);
#endif