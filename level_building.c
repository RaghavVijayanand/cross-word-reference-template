#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <dirent.h>

#define MAX_PATH_LEN 512
#define MAX_LEVELS 10
#define INF DBL_MAX
#define INS_PENALTY 200.0
#define DEL_PENALTY 200.0

typedef struct {
    int frames;
    int coeffs;
    double **data;
    char name[10]; 
} MFCC;

MFCC* load_mfcc(const char *filename, const char *name, int downsample) {
    FILE *f = fopen(filename, "r");
    if (!f) return NULL;

    int frames, coeffs;
    if (fscanf(f, "%d %d", &frames, &coeffs) != 2) {
        fclose(f); return NULL;
    }

    MFCC *mfcc = (MFCC*)malloc(sizeof(MFCC));
    mfcc->coeffs = coeffs;
    
    if (downsample) {
        mfcc->frames = (frames + 1) / 2;
    } else {
        mfcc->frames = frames;
    }

    mfcc->data = (double**)malloc(mfcc->frames * sizeof(double*));
    
    for (int i = 0; i < frames; i++) {
        double *temp_data = (double*)malloc(coeffs * sizeof(double));
        for (int j = 0; j < coeffs; j++) {
            fscanf(f, "%lf", &temp_data[j]);
        }
        
        if (downsample) {
            if (i % 2 == 0) { 
                mfcc->data[i/2] = temp_data;
            } else {
                free(temp_data);
            }
        } else {
            mfcc->data[i] = temp_data;
        }
    }
    if (name) strcpy(mfcc->name, name);
    fclose(f);
    return mfcc;
}

void free_mfcc(MFCC *mfcc) {
    if (!mfcc) return;
    for (int i = 0; i < mfcc->frames; i++) {
        free(mfcc->data[i]);
    }
    free(mfcc->data);
    free(mfcc);
}

double euclidean_dist(double *v1, double *v2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Global templates
#define MAX_TEMPLATES 100
MFCC *templates[MAX_TEMPLATES];
int num_templates = 0;

// Level Building Variables
double **D;
int **B;
int **W;

void save_debug_matrix(double **D, int L, int T, const char *filename);

void level_building(MFCC *test, const char *debug_name) {
    int T = test->frames;
    int L = 7; 

    D = (double**)malloc((L + 1) * sizeof(double*));
    B = (int**)malloc((L + 1) * sizeof(int*));
    W = (int**)malloc((L + 1) * sizeof(int*));
    for (int l = 0; l <= L; l++) {
        D[l] = (double*)malloc(T * sizeof(double));
        B[l] = (int*)malloc(T * sizeof(int));
        W[l] = (int*)malloc(T * sizeof(int));
        for (int t = 0; t < T; t++) {
            D[l][t] = INF;
            B[l][t] = -1;
            W[l][t] = -1;
        }
    }

    for (int l = 1; l <= L; l++) {
        for (int k = 0; k < num_templates; k++) {
            MFCC *ref = templates[k];
            int R = ref->frames;
            
            double **local_d = (double**)malloc(R * sizeof(double*));
            int **start_frame = (int**)malloc(R * sizeof(int*));
            
            for(int r=0; r<R; r++) {
                local_d[r] = (double*)malloc(T * sizeof(double));
                start_frame[r] = (int*)malloc(T * sizeof(int));
                for(int t=0; t<T; t++) {
                    local_d[r][t] = INF;
                    start_frame[r][t] = -1;
                }
            }
            
            // Initialization
            for (int t = 0; t < T; t++) {
                double cost = euclidean_dist(ref->data[0], test->data[t], ref->coeffs);
                double prev_cost = (l == 1) ? ((t == 0) ? 0.0 : INF) : ((t > 0) ? D[l-1][t-1] : INF);
                
                if (prev_cost != INF) {
                    local_d[0][t] = cost + prev_cost;
                    start_frame[0][t] = t; 
                }
            }
            
            // DP
            for (int t = 1; t < T; t++) {
                for (int r = 1; r < R; r++) {
                    double cost = euclidean_dist(ref->data[r], test->data[t], ref->coeffs);
                    
                    double v1 = local_d[r][t-1] + INS_PENALTY;     
                    double v2 = local_d[r-1][t-1];                 
                    double v3 = local_d[r-1][t] + DEL_PENALTY;     
                    
                    double min_v = v1;
                    int best_s = start_frame[r][t-1];
                    
                    if (v2 < min_v) {
                        min_v = v2;
                        best_s = start_frame[r-1][t-1];
                    }
                    if (v3 < min_v) { 
                        min_v = v3;
                        best_s = start_frame[r-1][t];
                    }
                    
                    if (min_v != INF) {
                        local_d[r][t] = min_v + cost;
                        start_frame[r][t] = best_s;
                    }
                }
            }
            
            // Update Global D[l][t]
            for (int t = 0; t < T; t++) {
                int duration = t - start_frame[R-1][t] + 1;
                int min_len = R / 2;
                if (min_len < 7) min_len = 7; // Force minimum 7 frames
                int max_len = R * 2;

                if (duration >= min_len && duration <= max_len) {
                    if (local_d[R-1][t] < D[l][t]) {
                        D[l][t] = local_d[R-1][t];
                        B[l][t] = start_frame[R-1][t];
                        W[l][t] = k;
                    }
                }
            }
            
            for(int r=0; r<R; r++) {
                free(local_d[r]);
                free(start_frame[r]);
            }
            free(local_d);
            free(start_frame);
        }
    }
    
    if (debug_name) {
        save_debug_matrix(D, L, T, debug_name);
    }
    
    double min_global = INF;
    int best_L = -1;
    
    for (int l = 1; l <= L; l++) {
        if (D[l][T-1] < min_global) {
            min_global = D[l][T-1];
            best_L = l;
        }
    }
    
    if (best_L != -1) {
        int curr_t = T - 1;
        int *result_digits = (int*)malloc(best_L * sizeof(int));
        
        for (int l = best_L; l >= 1; l--) {
            int k = W[l][curr_t];
            int start_t = B[l][curr_t];
            result_digits[l-1] = k;
            curr_t = start_t - 1; 
        }
        
        for (int i = 0; i < best_L; i++) {
            // Use the name from the template
            char name = templates[result_digits[i]]->name[0];
            if (name == 's') printf("_"); // Silence
            else printf("%c", name);
        }
        free(result_digits);
    } else {
        printf("No match");
    }
    
    for (int l = 0; l <= L; l++) {
        free(D[l]); free(B[l]); free(W[l]);
    }
    free(D); free(B); free(W);
}

void save_debug_matrix(double **D, int L, int T, const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;
    for (int l = 1; l <= L; l++) {
        for (int t = 0; t < T; t++) {
            double val = (D[l][t] == INF) ? -1.0 : D[l][t];
            fprintf(f, "%.4f ", val);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf(" [Saved debug_matrix.txt]");
}

int main() {
    struct dirent *de;
    DIR *dr = opendir("reference_templates_c");
    if (dr == NULL) {
        printf("Could not open directory reference_templates_c\n");
        return 1;
    }
    
    char path[MAX_PATH_LEN];
    while ((de = readdir(dr)) != NULL) {
        if (strstr(de->d_name, ".mfcc")) {
            sprintf(path, "reference_templates_c/%s", de->d_name);
            // Use filename as name (e.g., "1_fsdd.mfcc" -> "1")
            templates[num_templates] = load_mfcc(path, de->d_name, 0);
            if (templates[num_templates]) {
                num_templates++;
                if (num_templates >= MAX_TEMPLATES) break;
            }
        }
    }
    closedir(dr);
    printf("Loaded %d templates.\n", num_templates);

    dr = opendir("mfcc_connected/Team-1");
    if (dr == NULL) {
        printf("Could not open directory mfcc_connected/Team-1\n");
        return 1;
    }

    printf("Filename\tGroundTruth\tRecognized\n");
    printf("----------------------------------------\n");

    while ((de = readdir(dr)) != NULL) {
        if (strstr(de->d_name, ".mfcc")) {
            // Skip files containing 'o' or 'z'
            if (strchr(de->d_name, 'o') || strchr(de->d_name, 'z')) {
                continue;
            }

            sprintf(path, "mfcc_connected/Team-1/%s", de->d_name);
            MFCC *test = load_mfcc(path, NULL, 0); 
            if (test) {
                printf("%s\t", de->d_name);
                
                char gt[50];
                strcpy(gt, de->d_name);
                char *dot = strchr(gt, '.');
                if (dot) *dot = 0;
                for(int c=0; gt[c]; c++) {
                    if (strchr("123456789oz", gt[c])) printf("%c", gt[c]);
                }
                printf("\t\t");
                
                const char *debug_file = NULL;
                if (strcmp(de->d_name, "111a.mfcc") == 0) {
                    debug_file = "debug_matrix.txt";
                }
                
                level_building(test, debug_file);
                
                printf("\n");
                free_mfcc(test);
            }
        }
    }
    closedir(dr);
    return 0;
}
