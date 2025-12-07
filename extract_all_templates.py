import os
import glob
import librosa
import numpy as np

def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T

def save_mfcc(mfcc_data, path):
    with open(path, 'w') as f:
        f.write(f"{mfcc_data.shape[0]} {mfcc_data.shape[1]}\n")
        np.savetxt(f, mfcc_data, fmt='%.6f')

def main():
    wav_dir = "d:/e2e speech/project/Dataset/Connected_Digits/Dev/Team-1"
    out_dir = "reference_templates_c"
    # Clear existing templates to avoid mixing
    if os.path.exists(out_dir):
        for f in glob.glob(os.path.join(out_dir, "*.mfcc")):
            # Don't delete sil.mfcc if it exists, or just recreate it later
            if "sil.mfcc" not in f:
                os.remove(f)
    os.makedirs(out_dir, exist_ok=True)
    
    # Files to process (excluding o/z)
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
    
    print("Extracting ALL templates...")
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
            
            # Save every instance
            safe_fname = fname.replace('.', '_')
            out_name = f"{digit}_{safe_fname}_{i}.mfcc"
            out_path = os.path.join(out_dir, out_name)
            save_mfcc(mfcc, out_path)
            print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
