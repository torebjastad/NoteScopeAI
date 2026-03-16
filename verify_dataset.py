import sys
import os
import glob
import math
import numpy as np
import torch
import torchaudio
import librosa
import json
from tqdm import tqdm

# Add the user's PianoTunerPython project to sys.path
PIANO_TUNER_DIR = r"C:\Users\ToreGrüner\OneDrive\Code\GuitarTuner\PianoTunerPython"
sys.path.append(PIANO_TUNER_DIR)

try:
    from ultimate_detector import UltimatePianoDetector
except ImportError as e:
    print(f"Error importing UltimatePianoDetector: {e}")
    sys.exit(1)

from dataset import PianoToneDataset

def get_cents_tolerance(freq):
    # Dynamic tolerance based on piano inharmonicity curves
    if freq >= 2093: return 100  # C7+
    if freq >= 1047: return 50   # C6+
    if freq < 65.5:  return 40   # A0-C1 (high inharmonicity)
    return 40

def find_onset(data, sr, hop=1024, threshold_ratio=0.1):
    """RMS-based onset detection."""
    rms_vals = []
    for i in range(0, len(data) - hop, hop):
        chunk = data[i:i+hop]
        rms_vals.append(np.sqrt(np.mean(chunk**2)))
    if not rms_vals:
        return 0
    peak_rms = max(rms_vals)
    threshold = peak_rms * threshold_ratio
    for i, rms in enumerate(rms_vals):
        if rms > threshold:
            onset_sample = i * hop + int(sr * 0.05)  # 50ms past onset
            return min(onset_sample, len(data) - 8192)
    return 0

def index_to_expected_freq(index: int) -> float:
    """Calculate expected frequency in Hz for a piano key index (0-87)."""
    midi_note = index + 21
    # A4 is MIDI note 69, 440 Hz
    freq = 440.0 * (2 ** ((midi_note - 69) / 12))
    return freq

def get_cents_difference(f1: float, f2: float) -> float:
    """Calculate the difference in cents between two frequencies."""
    if f1 <= 0 or f2 <= 0:
        return float('inf')
    return 1200 * math.log2(f1 / f2)

def verify_dataset(data_dir: str, subset_filter: str = None):
    print(f"Initializing Dataset from {data_dir}...")
    dataset = PianoToneDataset(data_dir, augment=False)
    
    # Filter files if subset is requested
    if subset_filter:
        subset_filter = subset_filter.lower()
        filtered_files = []
        filtered_labels = []
        for f, lbl in zip(dataset.files, dataset.labels):
            if subset_filter in f.lower():
                filtered_files.append(f)
                filtered_labels.append(lbl)
        dataset.files = filtered_files
        dataset.labels = filtered_labels
        print(f"Filtered subset to '{subset_filter}': {len(dataset.files)} files")
    
    detector = UltimatePianoDetector()
    
    passed_files = []
    failed_files = []
    
    print(f"Starting verification of {len(dataset.files)} files...")
    
    for _ in range(3): print(".", end="", flush=True) # visual buffer
    
    for i in tqdm(range(len(dataset.files))):
        filepath = dataset.files[i]
        label_index = dataset.labels[i]
        
        expected_freq = index_to_expected_freq(label_index)
        
        try:
            # We load the full audio for pitch detection using librosa for simplicity and consistency with the tuner
            waveform, sample_rate = librosa.load(filepath, sr=None, mono=True)
            
            # Use onset detection from user's algorithm
            start_idx = find_onset(waveform, sample_rate)
            
            buffer_size = 8192
            if start_idx + buffer_size > len(waveform):
                start_idx = max(0, len(waveform) - buffer_size)
            buf = waveform[start_idx : start_idx + buffer_size]
            
            # The UltimatePianoDetector expects exactly this small numpy array buffer.
            detected_pitch = detector.get_pitch(buf, sample_rate)
            
            if detected_pitch <= 0:
                cents_diff = float('inf')
            else:
                cents_diff = get_cents_difference(detected_pitch, expected_freq)
                
            tolerance = get_cents_tolerance(expected_freq)
            
            octave_corrected_diff = abs(abs(cents_diff) - 1200)
            if abs(cents_diff) <= tolerance or octave_corrected_diff <= tolerance:
                passed_files.append({
                    "filepath": filepath,
                    "expected_freq": expected_freq,
                    "detected_freq": detected_pitch,
                    "cents_diff": cents_diff
                })
            else:
                failed_files.append({
                    "filepath": filepath,
                    "reason": "Pitch Mismatch" if detected_pitch > 0 else "Detection Failed",
                    "expected_freq": expected_freq,
                    "detected_freq": detected_pitch,
                    "cents_diff": cents_diff
                })
                
        except Exception as e:
            failed_files.append({
                "filepath": filepath,
                "reason": f"Error Processing: {str(e)}",
                "expected_freq": expected_freq,
                "detected_freq": -1,
                "cents_diff": float('inf')
            })

    print(f"\nVerification Complete.")
    print(f"Passed: {len(passed_files)}")
    print(f"Failed: {len(failed_files)}")
    
    # Save reports
    with open("verified_files.json", "w") as f:
        json.dump([p["filepath"] for p in passed_files], f, indent=4)
        
    with open("failed_files_report.txt", "w") as f:
        f.write(f"Failed Verification Report\n==========================\n")
        f.write(f"Total Failed: {len(failed_files)}\n\n")
        
        # Group by reason
        for fail in failed_files:
            f.write(f"File: {fail['filepath']}\n")
            f.write(f"  Reason: {fail['reason']}\n")
            if "Pitch Mismatch" in fail['reason']:
                f.write(f"  Expected: {fail['expected_freq']:.2f} Hz\n")
                f.write(f"  Detected: {fail['detected_freq']:.2f} Hz\n")
                f.write(f"  Diff: {fail['cents_diff']:.1f} cents\n")
            f.write("\n")
            
    print("Reports saved to verified_files.json and failed_files_report.txt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default=None, help="Filter to only test files containing this substring (e.g. 'uiowa')")
    args = parser.parse_args()
    
    data_directory = r"C:\Users\ToreGrüner\SkyDrive\Dokumenter\Musikk\Piano"
    verify_dataset(data_directory, subset_filter=args.subset)
