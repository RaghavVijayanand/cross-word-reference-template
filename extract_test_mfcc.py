import os
import glob
import librosa
import numpy as np

def save_mfcc(mfcc_data, path):
    with open(path, 'w') as f:
        f.write(f"{mfcc_data.shape[0]} {mfcc_data.shape[1]}\n")
        np.savetxt(f, mfcc_data, fmt='%.6f')

def main():
    wav_dir = "d:/e2e speech/project/Dataset/Connected_Digits/Dev/Team-1"
    out_dir = "mfcc_connected/Team-1"
    os.makedirs(out_dir, exist_ok=True)
    
    # Process all .wav files in the directory
    wav_files = glob.glob(os.path.join(wav_dir, "*.wav"))
    
    print(f"Found {len(wav_files)} wav files in {wav_dir}")
    
    for wav_path in wav_files:
        filename = os.path.basename(wav_path)
        # Skip files with 'o' or 'z' if we want to be consistent with previous filtering, 
        # but let's convert everything and let the C code filter.
        
        y, sr = librosa.load(wav_path, sr=None)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = mfcc.T
        
        out_name = filename.replace('.wav', '.mfcc')
        out_path = os.path.join(out_dir, out_name)
        
        save_mfcc(mfcc, out_path)
        print(f"Converted {filename} -> {out_name}")

if __name__ == "__main__":
    main()
