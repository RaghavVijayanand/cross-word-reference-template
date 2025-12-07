import os
import librosa
import numpy as np

def save_mfcc(mfcc_data, path):
    with open(path, 'w') as f:
        f.write(f"{mfcc_data.shape[0]} {mfcc_data.shape[1]}\n")
        np.savetxt(f, mfcc_data, fmt='%.6f')

def main():
    # Use 11a.wav as source for silence
    wav_path = "d:/e2e speech/project/Dataset/Connected_Digits/Dev/Team-1/11a.wav"
    out_path = "reference_templates_c/sil.mfcc"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    y, sr = librosa.load(wav_path, sr=None)
    
    # Take the first 0.15 seconds (approx 15 frames) as silence
    # Before trimming!
    silence_duration = 0.15
    num_samples = int(silence_duration * sr)
    y_sil = y[:num_samples]
    
    mfcc = librosa.feature.mfcc(y=y_sil, sr=sr, n_mfcc=13)
    mfcc = mfcc.T
    
    save_mfcc(mfcc, out_path)
    print(f"Extracted silence template to {out_path} ({mfcc.shape[0]} frames)")

if __name__ == "__main__":
    main()
