"""
Preprocessing pipeline: converts verified audio files into pre-computed mel
spectrogram tensors on disk. This runs once and makes training ~10-20x faster
by eliminating librosa (slow) from the training loop entirely.

Output layout:
    preprocessed/
        manifest.json          # [{path, label}, ...]
        specs/
            0000_label45_s0.pt  # mel spec for file 0, shift=0
            0000_label45_s1.pt  # mel spec for file 0, shift=+1
            ...
"""

import os
import json
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_SR     = 22050
N_SAMPLES     = 8192
N_FFT         = 1024
HOP_LENGTH    = 256
N_MELS        = 128
TOP_DB        = 30        # silence trim threshold
PITCH_SHIFTS  = [-2, -1, 0, 1, 2]   # semitones

VERIFIED_JSON = "verified_files.json"
OUT_DIR       = "preprocessed"
SPECS_DIR     = os.path.join(OUT_DIR, "specs")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")
# ──────────────────────────────────────────────────────────────────────────────

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
amp_to_db = T.AmplitudeToDB()


def load_and_trim(filepath: str) -> torch.Tensor:
    """Load audio, resample to TARGET_SR, mix to mono, trim silence."""
    waveform, sr = torchaudio.load(filepath)

    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        waveform = F.resample(waveform, sr, TARGET_SR)

    # Trim silence using librosa (done once, OK to be slow here)
    wav_np = waveform.numpy()[0]
    trimmed_np, _ = librosa.effects.trim(wav_np, top_db=TOP_DB)
    waveform = torch.from_numpy(trimmed_np).unsqueeze(0)

    return waveform


def pad_or_crop(waveform: torch.Tensor) -> torch.Tensor:
    length = waveform.shape[1]
    if length >= N_SAMPLES:
        return waveform[:, :N_SAMPLES]
    padding = N_SAMPLES - length
    return torch.nn.functional.pad(waveform, (0, padding))


def to_mel_spec(waveform: torch.Tensor) -> torch.Tensor:
    spec = mel_transform(waveform)
    spec = amp_to_db(spec)
    return spec


def apply_pitch_shift(waveform: torch.Tensor, n_steps: int) -> torch.Tensor:
    if n_steps == 0:
        return waveform
    return F.pitch_shift(waveform, TARGET_SR, n_steps)


def label_from_filepath(filepath: str) -> int:
    """Re-derive label from filename (same logic as dataset.py)."""
    import re
    filename = os.path.basename(filepath)
    note_to_semitone = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    match = re.search(r"([A-G][b#]?\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse label from: {filename}")
    note_str = match.group(1)
    m = re.match(r"([A-G][b#]?)(-?\d+)", note_str)
    if not m:
        raise ValueError(f"Cannot parse note: {note_str}")
    pitch_class, octave = m.groups()
    midi_note = (int(octave) + 1) * 12 + note_to_semitone[pitch_class]
    index = midi_note - 21
    if index < 0 or index > 87:
        raise ValueError(f"Out of range: {note_str}")
    return index


def main():
    os.makedirs(SPECS_DIR, exist_ok=True)

    with open(VERIFIED_JSON, 'r') as f:
        verified_files = json.load(f)

    print(f"Preprocessing {len(verified_files)} verified files...")
    print(f"Pitch shift variants: {PITCH_SHIFTS}")
    print(f"Output: {OUT_DIR}/")

    manifest = []
    errors = 0

    for file_idx, filepath in enumerate(tqdm(verified_files)):
        try:
            label = label_from_filepath(filepath)
        except ValueError as e:
            print(f"\nSkipping (label error): {e}")
            errors += 1
            continue

        try:
            waveform = load_and_trim(filepath)
            waveform = pad_or_crop(waveform)
        except Exception as e:
            print(f"\nSkipping (load error) {filepath}: {e}")
            errors += 1
            continue

        for shift in PITCH_SHIFTS:
            shifted_label = label + shift
            if shifted_label < 0 or shifted_label > 87:
                continue  # out of piano range

            try:
                shifted_wav = apply_pitch_shift(waveform, shift)
                spec = to_mel_spec(shifted_wav)

                shift_tag = f"p{shift}" if shift >= 0 else f"m{abs(shift)}"
                out_filename = f"{file_idx:04d}_lbl{label:02d}_{shift_tag}.pt"
                out_path = os.path.join(SPECS_DIR, out_filename)

                torch.save(spec, out_path)
                manifest.append({"path": out_path, "label": shifted_label})

            except Exception as e:
                print(f"\nSkipping shift={shift} for {filepath}: {e}")
                errors += 1

    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f)

    print(f"\nDone. {len(manifest)} spectrogram files written, {errors} errors.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
