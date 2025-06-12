#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ROWS 10000
#define MAX_FEATURES 200

float features[MAX_ROWS][MAX_FEATURES];
int labels[MAX_ROWS];

void load_csv(const char* filename, int* num_rows_out) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Dosya açılamadı");
        exit(EXIT_FAILURE);
    }

    char line[2048];
    int row = 0;

    
    fgets(line, sizeof(line), file);

    while (fgets(line, sizeof(line), file) && row < MAX_ROWS) {
        char* token = strtok(line, ","); // ID_code
        token = strtok(NULL, ",");       // target
        labels[row] = atoi(token);

        for (int i = 0; i < MAX_FEATURES; i++) {
            token = strtok(NULL, ",");
            if (token != NULL)
                features[row][i] = strtof(token, NULL);
            else
                features[row][i] = 0.0f;
        }

        row++;
    }

    *num_rows_out = row;
    fclose(file);
}