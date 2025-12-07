import numpy as np
import sys
import os

def read_mfcc(filename):
    """Reads MFCC data from a file."""
    try:
        with open(filename, 'r') as f:
            header = f.readline().split()
            if len(header) != 2:
                return None, None
            frames, dimension = int(header[0]), int(header[1])
            data = []
            for _ in range(frames):
                line = f.readline().strip().split()
                data.append([float(x) for x in line])
            data = np.array(data)
            # Cepstral Mean Normalization (CMN)
            mean = np.mean(data, axis=0)
            data = data - mean
            return data, dimension
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None

def calculate_distance_matrix(test_data, template_data):
    """Calculates Euclidean distance matrix between test and template."""
    # test: (T, D), template: (R, D)
    # dist: (T, R)
    # Using broadcasting: (T, 1, D) - (1, R, D) -> (T, R, D) -> norm -> (T, R)
    diff = test_data[:, np.newaxis, :] - template_data[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist

def level_building(test_data, templates):
    """
    Performs Level Building algorithm.
    test_data: (T, D)
    templates: dict of label -> (R, D)
    """
    T = test_data.shape[0]
    labels = list(templates.keys())
    num_templates = len(labels)
    
    # Parameters
    MAX_LEVELS = 10 # Maximum number of digits expected
    INF = 1e9
    
    # Best score to reach time t at level l
    # best_score[l, t]
    best_score = np.full((MAX_LEVELS + 1, T), INF)
    
    # Backpointers
    # backptr_word[l, t]: index of word ending at t at level l
    backptr_word = np.full((MAX_LEVELS + 1, T), -1, dtype=int)
    # backptr_time[l, t]: start time of word ending at t at level l (actually end of prev word)
    backptr_time = np.full((MAX_LEVELS + 1, T), -1, dtype=int)
    
    # Initialization
    # Level 0: Cost 0 at time -1 (conceptually). 
    # We handle level 1 separately or initialize level 0 boundary.
    # Let's say level 0 ends at time -1 with cost 0.
    
    # To implement this efficiently, we iterate levels.
    # For Level 1: Start at t=0.
    # For Level L: Start at t where Level L-1 ended.
    
    # Pre-compute distance matrices for all templates
    dist_matrices = {}
    for label in labels:
        dist_matrices[label] = calculate_distance_matrix(test_data, templates[label])
        
    # Initialization for Level 1
    # We can start at t=0.
    # We run DTW for each template starting at t=0.
    
    # Actually, standard Level Building computes "best path for template k starting at t-1 and ending at t"
    # But that's expensive (run DTW for every start/end pair).
    # The "One-Pass" equivalent is better:
    # At each time t, we update the state of all templates.
    # But we need to track levels if we want fixed length or just "best sequence".
    # User asked for "continuous detection", usually implies unknown length.
    # So "One-Pass" (Viterbi-like) is appropriate.
    
    # One-Pass State:
    # cost[k, j]: Cost to reach frame j of template k at current time t
    # We also need to track the "history" (best previous word and its end time) for each state?
    # No, usually we just track the "token" that enters the start of the word.
    
    # Let's stick to the One-Pass approach I tried in C, but cleaner.
    
    # Global Best Path to reach time t (end of a word)
    global_best_score = np.full(T, INF)
    global_backptr_word = np.full(T, -1, dtype=int)
    global_backptr_time = np.full(T, -1, dtype=int)
    
    # Active templates state
    # state_cost[k, j]
    state_cost = []
    state_start_time = [] # When did this word start?
    
    for label in labels:
        R = templates[label].shape[0]
        state_cost.append(np.full(R, INF))
        state_start_time.append(np.full(R, -1, dtype=int))
        
    prev_global_best_score = 0.0 # Cost at t=-1
    
    for t in range(T):
        new_state_cost = []
        new_state_start_time = []
        
        # Best score to enter a new word at this time t
        # It comes from the best completed word at t-1
        # Word Insertion Penalty (WIP): Negative value encourages more words
        WIP = -5000.0
        entry_cost = prev_global_best_score + WIP
        entry_start_time = t
        
        current_global_best = INF
        current_best_word = -1
        current_best_time = -1
        
        for k, label in enumerate(labels):
            R = templates[label].shape[0]
            dist = dist_matrices[label][t] # Vector of size R
            
            curr_cost = state_cost[k]
            curr_start = state_start_time[k]
            
            next_cost = np.full(R, INF)
            next_start = np.full(R, -1, dtype=int)
            
            # Vectorized DP update
            # 1. Internal transitions
            # From (j) -> (j) [0]
            # From (j-1) -> (j) [1]
            # From (j-2) -> (j) [2]
            
            # We need to compute min of previous states
            # Construct a padded version for easy shifting
            padded_cost = np.pad(curr_cost, (2, 0), constant_values=INF)
            padded_start = np.pad(curr_start, (2, 0), constant_values=-1)
            
            # Candidates for each j:
            # j (from j in prev): padded_cost[2:]
            # j-1 (from j-1 in prev): padded_cost[1:-1]
            # j-2 (from j-2 in prev): padded_cost[:-2]
            
            c0 = padded_cost[2:]
            c1 = padded_cost[1:-1]
            c2 = padded_cost[:-2]
            
            # Find min cost
            min_prev = np.minimum(c0, np.minimum(c1, c2))
            
            # Determine which one was picked to propagate start_time
            # This is a bit tricky vectorized, let's do argmin
            stacked_costs = np.stack([c0, c1, c2]) # 3 x R
            choices = np.argmin(stacked_costs, axis=0) # R
            
            # Get corresponding start times
            # 0 -> padded_start[2:] (same j)
            # 1 -> padded_start[1:-1] (j-1)
            # 2 -> padded_start[:-2] (j-2)
            
            s0 = padded_start[2:]
            s1 = padded_start[1:-1]
            s2 = padded_start[:-2]
            stacked_starts = np.stack([s0, s1, s2])
            
            # Select best start times
            best_starts = np.choose(choices, stacked_starts)
            
            # 2. Entry transition at j=0
            # If entry_cost < min_prev[0], we take entry
            if entry_cost < min_prev[0]:
                min_prev[0] = entry_cost
                best_starts[0] = entry_start_time
                
            # Update costs
            next_cost = min_prev + dist
            next_start = best_starts
            
            new_state_cost.append(next_cost)
            new_state_start_time.append(next_start)
            
            # Check exit (last frame)
            if next_cost[-1] < current_global_best:
                current_global_best = next_cost[-1]
                current_best_word = k
                current_best_time = next_start[-1] - 1 # End of prev word
                
        state_cost = new_state_cost
        state_start_time = new_state_start_time
        
        global_best_score[t] = current_global_best
        global_backptr_word[t] = current_best_word
        global_backptr_time[t] = current_best_time
        
        prev_global_best_score = current_global_best

    # Backtracking
    result = []
    curr_t = T - 1
    
    # Optional: Find best ending in last few frames
    # best_t = np.argmin(global_best_score[T-5:T]) + (T-5)
    # curr_t = best_t
    
    while curr_t >= 0:
        word_idx = global_backptr_word[curr_t]
        if word_idx == -1:
            break
        result.append(labels[word_idx])
        curr_t = global_backptr_time[curr_t]
        
    return result[::-1]

def main():
    if len(sys.argv) < 2:
        print("Usage: python recognizer.py <test_mfcc>")
        sys.exit(1)
        
    test_file = sys.argv[1]
    
    # Load templates
    templates = {}
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "o", "z"]
    
    template_dir = "reference_templates_c"
    for l in labels:
        # Try using the _1 variant if available, else fallback
        path = os.path.join(template_dir, f"{l}_1.mfcc")
        if not os.path.exists(path):
             path = os.path.join(template_dir, f"{l}.mfcc")
             
        data, dim = read_mfcc(path)
        if data is not None:
            templates[l] = data
            
    test_data, dim = read_mfcc(test_file)
    if test_data is None:
        print("Error reading test file")
        sys.exit(1)
        
    result = level_building(test_data, templates)
    print("Recognition Result:", " ".join(result))

if __name__ == "__main__":
    main()
