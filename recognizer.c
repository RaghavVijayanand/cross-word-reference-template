#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_FRAMES 2000
#define MAX_DIMENSION 39
#define MAX_TEMPLATES 15
#define INF 1e9

typedef struct {
    int frames;
    int dimension;
    double **data;
    char label[10];
} MFCC;

MFCC *read_mfcc(const char *filename, const char *label) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    int frames, dimension;
    if (fscanf(fp, "%d %d", &frames, &dimension) != 2) {
        printf("Error reading header from %s\n", filename);
        fclose(fp);
        return NULL;
    }

    MFCC *mfcc = (MFCC *)malloc(sizeof(MFCC));
    mfcc->frames = frames;
    mfcc->dimension = dimension;
    strncpy(mfcc->label, label, 9);
    mfcc->label[9] = '\0';

    mfcc->data = (double **)malloc(frames * sizeof(double *));
    for (int i = 0; i < frames; i++) {
        mfcc->data[i] = (double *)malloc(dimension * sizeof(double));
        for (int j = 0; j < dimension; j++) {
            fscanf(fp, "%lf", &mfcc->data[i][j]);
        }
    }

    fclose(fp);
    return mfcc;
}

double calculate_distance(double *v1, double *v2, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// One-Pass DP for Continuous Digit Recognition
void recognize(MFCC *test, MFCC **templates, int num_templates) {
    int T = test->frames;
    
    // best_score[t]: Minimum cost to reach time t having completed a word
    double *best_score = (double *)malloc(T * sizeof(double));
    // backptr_word[t]: Which word ended at time t
    int *backptr_word = (int *)malloc(T * sizeof(int));
    // backptr_time[t]: When did that word start (or previous word end)
    int *backptr_time = (int *)malloc(T * sizeof(int));

    for (int t = 0; t < T; t++) {
        best_score[t] = INF;
        backptr_word[t] = -1;
        backptr_time[t] = -1;
    }

    // Internal DP state for each template: current_cost[template][template_frame]
    double **current_cost = (double **)malloc(num_templates * sizeof(double *));
    double **next_cost = (double **)malloc(num_templates * sizeof(double *));

    for (int k = 0; k < num_templates; k++) {
        int ref_frames = templates[k]->frames;
        current_cost[k] = (double *)malloc(ref_frames * sizeof(double));
        next_cost[k] = (double *)malloc(ref_frames * sizeof(double));
        for (int j = 0; j < ref_frames; j++) {
            current_cost[k][j] = INF;
        }
    }

    // Initialization: At time -1, score is 0
    double prev_best_score = 0.0;

    for (int t = 0; t < T; t++) {
        // For each template
        for (int k = 0; k < num_templates; k++) {
            int ref_frames = templates[k]->frames;
            
            for (int j = 0; j < ref_frames; j++) {
                double dist = calculate_distance(test->data[t], templates[k]->data[j], test->dimension);
                
                double min_prev = INF;
                
                // Transitions
                // 1. From same template, previous frame (j-1)
                if (j > 0) {
                    if (current_cost[k][j-1] < min_prev) min_prev = current_cost[k][j-1];
                }
                // 2. From same template, same frame (j) - Self loop (optional, often skipped in simple DTW or penalized)
                // Let's stick to simple Itakura-like: (j-1, t-1), (j, t-1), (j-2, t-1)
                // But here we are iterating t. So we look at current_cost (which is t-1).
                
                if (current_cost[k][j] < min_prev) min_prev = current_cost[k][j]; // Horizontal (j, t-1)
                if (j > 1) {
                     if (current_cost[k][j-2] < min_prev) min_prev = current_cost[k][j-2];
                }

                // 3. Entry from completed word (j=0 only)
                if (j == 0) {
                    // If we are starting the template, we can come from the best score at t-1
                    // But wait, best_score[t-1] represents a completed word at t-1.
                    // So we start matching frame 0 at time t.
                    if (prev_best_score < min_prev) min_prev = prev_best_score;
                }

                next_cost[k][j] = min_prev + dist;
            }

            // Check if this template ended
            double exit_cost = next_cost[k][ref_frames - 1];
            if (exit_cost < best_score[t]) {
                best_score[t] = exit_cost;
                backptr_word[t] = k;
                // We don't strictly know start time here without propagating it, 
                // but for simple reconstruction we can just track the sequence.
                // To track duration, we'd need to propagate 'start_time' through the template.
                // For now, let's just track that a word ended here.
            }
        }

        // Update current_cost for next time step
        for (int k = 0; k < num_templates; k++) {
            for (int j = 0; j < templates[k]->frames; j++) {
                current_cost[k][j] = next_cost[k][j];
            }
        }
        
        // Update prev_best_score for next step (t -> t+1 entry)
        prev_best_score = best_score[t];
    }

    // Backtracking
    // We look for the best ending at the final frame T-1, or nearby?
    // Usually the utterance ends with silence or the last digit.
    // We backtrack from T-1.
    
    // Since we didn't propagate start times, we have to infer the path or use a simpler greedy approach?
    // Actually, without propagating start times (or backpointers within the template), 
    // we can't easily jump back to the start of the word.
    // FIX: We need to propagate the "start time" or "duration" through the template grid.
    
    // Let's re-implement with a simpler structure:
    // Just find the sequence.
    // If we only store "best_score[t]" and "backptr_word[t]", we know WHICH word ended at t.
    // But we don't know WHERE it started, so we don't know where to look for the previous word.
    // We MUST propagate the duration or start time.
}

// Re-implementation with duration propagation
void recognize_corrected(MFCC *test, MFCC **templates, int num_templates) {
    int T = test->frames;
    
    // Global best path to reach time t (completed word)
    double *best_score = (double *)malloc(T * sizeof(double));
    int *word_history = (int *)malloc(T * sizeof(int)); // Which word ended at t
    int *time_history = (int *)malloc(T * sizeof(int)); // Where did the previous word end

    for (int t = 0; t < T; t++) {
        best_score[t] = INF;
        word_history[t] = -1;
        time_history[t] = -1;
    }

    // DP State: [template][frame] -> {cost, start_time}
    // We need two buffers: current (t-1) and next (t)
    typedef struct {
        double cost;
        int start_time;
    } State;

    State **current_state = (State **)malloc(num_templates * sizeof(State *));
    State **next_state = (State **)malloc(num_templates * sizeof(State *));

    for (int k = 0; k < num_templates; k++) {
        int frames = templates[k]->frames;
        current_state[k] = (State *)malloc(frames * sizeof(State));
        next_state[k] = (State *)malloc(frames * sizeof(State));
        for (int j = 0; j < frames; j++) {
            current_state[k][j].cost = INF;
            current_state[k][j].start_time = -1;
        }
    }

    // Initialization
    double prev_global_best = 0.0; // Cost at t=-1
    int prev_global_end = -1;      // End time of previous word

    // Word Insertion Penalty
    double WIP = 0.0; 

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < num_templates; k++) {
            int ref_frames = templates[k]->frames;
            
            for (int j = 0; j < ref_frames; j++) {
                double dist = calculate_distance(test->data[t], templates[k]->data[j], test->dimension);
                
                double min_cost = INF;
                int best_start = -1;

                // 1. From j-1 (Horizontal/Diagonal in grid, but time advances)
                
                // From (t-1, j)
                if (current_state[k][j].cost < min_cost) {
                    min_cost = current_state[k][j].cost;
                    best_start = current_state[k][j].start_time;
                }
                
                // From (t-1, j-1)
                if (j > 0 && current_state[k][j-1].cost < min_cost) {
                    min_cost = current_state[k][j-1].cost;
                    best_start = current_state[k][j-1].start_time;
                }

                // From (t-1, j-2)
                if (j > 1 && current_state[k][j-2].cost < min_cost) {
                    min_cost = current_state[k][j-2].cost;
                    best_start = current_state[k][j-2].start_time;
                }

                // 2. Entry (j=0)
                if (j == 0) {
                    // Can enter from a completed word at t-1
                    if (prev_global_best + WIP < min_cost) {
                        min_cost = prev_global_best + WIP;
                        best_start = t; 
                    }
                }

                next_state[k][j].cost = min_cost + dist;
                next_state[k][j].start_time = best_start;
            }

            // Check exit
            int last = ref_frames - 1;
            if (next_state[k][last].cost < best_score[t]) {
                best_score[t] = next_state[k][last].cost;
                word_history[t] = k;
                time_history[t] = next_state[k][last].start_time - 1;
            }
        }

        // Update buffers
        for (int k = 0; k < num_templates; k++) {
            for (int j = 0; j < templates[k]->frames; j++) {
                current_state[k][j] = next_state[k][j];
            }
        }
        
        prev_global_best = best_score[t];
    }

    // Backtrack
    printf("Recognition Result: ");
    int curr_t = T - 1;

    int result_stack[1000];
    int stack_idx = 0;

    while (curr_t >= 0) {
        int word_idx = word_history[curr_t];
        if (word_idx == -1) break; 
        
        result_stack[stack_idx++] = word_idx;
        curr_t = time_history[curr_t];
    }

    for (int i = stack_idx - 1; i >= 0; i--) {
        printf("%s ", templates[result_stack[i]]->label);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <test_mfcc>\n", argv[0]);
        return 1;
    }

    // Load templates
    MFCC *templates[MAX_TEMPLATES];
    int num_templates = 0;
    
    char *labels[] = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "o", "z"};
    char filepath[256];

    for (int i = 0; i < 11; i++) {
        sprintf(filepath, "reference_templates_c/%s.mfcc", labels[i]);
        MFCC *t = read_mfcc(filepath, labels[i]);
        if (t) {
            templates[num_templates++] = t;
        } else {
            printf("Warning: Could not load template %s\n", filepath);
        }
    }

    if (num_templates == 0) {
        printf("No templates loaded.\n");
        return 1;
    }

    // Load test file
    MFCC *test = read_mfcc(argv[1], "test");
    if (!test) {
        return 1;
    }

    recognize_corrected(test, templates, num_templates);

    return 0;
}
