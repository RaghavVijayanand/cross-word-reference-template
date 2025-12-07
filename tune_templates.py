import os
import glob
import librosa
import numpy as np
from scipy.spatial.distance import cdist

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def dtw_distance(s1, s2):
    # Simple DTW implementation for Python
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[:] = np.inf
    dtw_matrix[0, 0] = 0
    
    dist = cdist(s1, s2, metric='euclidean')
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist[i-1, j-1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                          dtw_matrix[i, j-1],    # deletion
                                          dtw_matrix[i-1, j-1])  # match
    return dtw_matrix[n, m]

def save_mfcc(mfcc_data, path):
    with open(path, 'w') as f:
        f.write(f"{mfcc_data.shape[0]} {mfcc_data.shape[1]}\n")
        np.savetxt(f, mfcc_data, fmt='%.6f')

def main():
    wav_dir = "d:/e2e speech/project/Dataset/Connected_Digits/Dev/Team-1"
    out_dir = "reference_templates_c"
    
    # Clear existing digit templates (keep silence if possible, but safer to clear all and re-run extract_silence)
    # Actually, let's just delete everything to be clean.
    if os.path.exists(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*.mfcc")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    
    files = {
        '11a.wav': ['1', '1'],
        '14a.wav': ['1', '4'],
        '15a.wav': ['1', '5'],
        '41a.wav': ['4', '1'],
        '44a.wav': ['4', '4'],
        '455a.wav': ['4', '5', '5'],
        '47a.wav': ['4', '7'],
        '545a.wav': ['5', '4', '5'],
        '55a.wav': ['5', '5'],
        '711a.wav': ['7', '1', '1'],
        '71a.wav': ['7', '1'],
        '74a.wav': ['7', '4'],
        '77a.wav': ['7', '7']
    }
    
    candidates = {'1': [], '4': [], '5': [], '7': []}
    
    print("Extracting candidates...")
    for fname, digits in files.items():
        path = os.path.join(wav_dir, fname)
        if not os.path.exists(path):
            continue
            
        y, sr = librosa.load(path, sr=None)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Naive split
        total_len = len(y)
        chunk_len = total_len // len(digits)
        
        for i, digit in enumerate(digits):
            start = i * chunk_len
            end = (i + 1) * chunk_len
            y_chunk = y[start:end]
            
            mfcc = extract_mfcc(y_chunk, sr)
            candidates[digit].append({
                'data': mfcc,
                'source': f"{fname}_pos{i}"
            })

    print("\nFinding Medoids (Single Best Template)...")
    for digit, items in candidates.items():
        if not items:
            continue
            
        n = len(items)
        dist_matrix = np.zeros((n, n))
        
        # Compute pairwise DTW
        for i in range(n):
            for j in range(i+1, n):
                d = dtw_distance(items[i]['data'], items[j]['data'])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        # Find medoid
        avg_dists = np.mean(dist_matrix, axis=1)
        best_idx = np.argmin(avg_dists)
        best_item = items[best_idx]
        
        print(f"Digit {digit}: Selected {best_item['source']} (Avg Dist: {avg_dists[best_idx]:.2f})")
        
        # Save Medoid as the ONLY template for this digit
        out_path = os.path.join(out_dir, f"{digit}.mfcc")
        save_mfcc(best_item['data'], out_path)

if __name__ == "__main__":
    main()
