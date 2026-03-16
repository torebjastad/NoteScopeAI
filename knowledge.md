# Piano Tone Recognition — Knowledge Base

Technical knowledge, design decisions, and lessons learned building this pipeline.

---

## 1. Project Overview

A deep learning pipeline that classifies piano notes (88 keys, A0–C8) from short audio clips.
Input: any audio file. Output: predicted note name + confidence.

**Current accuracy: 98.23% validation / 97.89% test** (as of 2026-03-15).

The pipeline has three phases:
1. **Preprocessing** — convert raw audio to mel spectrogram tensors stored on disk
2. **Training** — supervised CNN classification on those tensors
3. **Inference** — multi-window averaging for robust real-time prediction

---

## 2. Audio Representation

### Mel Spectrogram
Audio is converted to a 2D mel spectrogram tensor `[1, 128, T]`:
- 128 mel frequency bands
- T time frames (T ≈ 65 for a 16384-sample window at hop=256)
- Values in dB (log-amplitude) via `AmplitudeToDB`

Why mel scale: it matches human pitch perception logarithmically, so equal intervals on the y-axis correspond to equal perceptual pitch differences.

Why log-amplitude: the dynamic range of piano audio spans ~60 dB; log scale prevents quiet harmonics from being invisible.

### Window Size: 16384 samples (~743 ms at 22050 Hz)
- Captures the full attack + initial decay, which contain the most pitch-defining harmonic content
- Large enough to see multiple harmonic overtones clearly in the mel spectrogram
- Previous size was 8192 (~370 ms) — doubling it improved accuracy from 82.7% → 98.2%

### Chunking with 50% Overlap
Long files (especially UIowa samples, 30–40s) are split into multiple overlapping 16384-sample windows with 8192-sample (50%) hop. Benefits:
- More training samples per source file (avg ~15 chunks/file)
- Trains the model on the attack, sustain, and decay phases independently
- Inference averages all windows for robust prediction

### Silence Trimming
Before chunking, silence is trimmed using an RMS-based approach (`trim_silence_torch`):
- Computes RMS per frame via `tensor.unfold()`
- Removes leading/trailing frames below 30 dB threshold relative to peak
- GPU-compatible; validated equivalent to `librosa.effects.trim` (MAE < 0.001 dB)

---

## 3. Preprocessing Pipeline (preprocess_v2.py)

Two-stage GPU-accelerated pipeline:

### Stage A — Pitch shift → OGG
```
verified audio files
    → resample to 22050 Hz
    → silence trim (GPU, RMS-based)
    → pitch shift ×5 variants: −2, −1, 0, +1, +2 semitones (GPU)
    → save as OGG to processed_audio/{note_name}/shift_{tag}/{stem}.ogg
    → write processed_audio/manifest_a.json
```
Output: ~5371 OGG files

**Key constraint**: torchaudio's OGG encoder crashes on audio longer than ~25 seconds.
**Fix**: `try_save_ogg()` truncates to 20 seconds (440100 samples) before saving.

**Resume support**: Stage A checks if the output OGG already exists before processing.
If interrupted, re-run and it picks up where it left off.

### Stage B — Chunks → mel spectrograms
```
OGG files (from manifest_a.json)
    → load + resample
    → GPU
    → extract overlapping 16384-sample chunks (50% overlap, RMS filter)
    → GPU mel spectrogram + dB conversion
    → save .pt tensor to preprocessed_v2/specs/{idx}_lbl{label}_c{chunk}.pt
    → write preprocessed_v2/manifest.json
```
Output: 81,147 `.pt` tensor files

### Running the pipeline
```bash
# Full pipeline (validate first, then A+B)
python preprocess_v2.py

# Stage A only (with resume, skip validation)
python preprocess_v2.py --stage a --skip-validate

# Stage B only
python preprocess_v2.py --stage b --skip-validate

# Validate GPU vs CPU equivalence on 20 files
python preprocess_v2.py --validate
```

**Important**: Do NOT pipe the script output to `head` or `grep`. tqdm uses `\r` (carriage return) for progress updates — piping causes SIGPIPE which silently kills the Python process.

---

## 4. Dataset & Augmentation

### Source data
- 1088 verified audio files from `C:\Users\ToreGrüner\SkyDrive\Dokumenter\Musikk\Piano`
- Collections: Bechstein 1911 Upright (multi-velocity), UIowa Piano-mf, and others
- 188 files excluded due to pitch verification failure (stored in `failed_files_report.txt`)

### Pitch verification
Uses `UltimatePianoDetector` from the GuitarTuner project:
- Tolerances: ±40 cents (default/low notes), ±50 cents (C6+), ±100 cents (C7+)
- Octave errors accepted: if detected pitch is ~1200 cents off, the filename is ground truth
  (The detector sometimes identifies the 1st harmonic instead of the fundamental)

### Pitch shifting augmentation
- ±2 semitones maximum — beyond ±2 semitones, piano body resonances (formants) become
  unrealistic, making pitch-shifted samples sound synthetic
- GPU-accelerated via `torchaudio.functional.pitch_shift`

### Runtime augmentation (during training only)
Applied by `PreprocessedPianoDataset` when `augment=True`:
- Random gain: multiply spectrogram by uniform(0.5, 2.0)
- Random noise: add Gaussian noise with amplitude uniform(0, 0.005)
- These are cheap tensor operations — safe with `num_workers=4`

### Train/Val/Test split
80/10/10 split with seed=42, using index shuffling:
- Train subset uses `full_dataset` (augment=True)
- Val and Test subsets use `clean_dataset` (augment=False)
- Same indices applied to both → perfectly consistent splits

---

## 5. Model Architecture (PianoNet)

```
Input: [batch, 1, 128, ~65]   ← mel spectrogram

Conv2d(1→32, 3×3) + BN + ReLU + MaxPool2d(2×2)   → [batch, 32, 64, ~32]
Conv2d(32→64, 3×3) + BN + ReLU + MaxPool2d(2×2)  → [batch, 64, 32, ~16]
Conv2d(64→128, 3×3) + BN + ReLU + MaxPool2d(2×2) → [batch, 128, 16, ~8]

AdaptiveAvgPool2d(8×4)  → [batch, 128, 8, 4]   ← fixed size regardless of input length
Flatten                 → [batch, 4096]

Dropout(0.3)
Linear(4096→512) + ReLU
Dropout(0.3)
Linear(512→88)          ← one logit per piano key
```

`AdaptiveAvgPool2d` is the key design choice — it allows the model to accept variable-length spectrograms (both during training on 16384-sample chunks and inference on arbitrarily long files).

---

## 6. Training (train.py)

- **Optimizer**: Adam, lr=0.001
- **Loss**: CrossEntropyLoss
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3) — halves LR when val loss stalls
- **Early stopping**: patience=8 epochs with no val accuracy improvement
- **Best model checkpoint**: saves `piano_net_best.pth` whenever val accuracy improves
- **Final model**: at end of training, loads best checkpoint and saves as `piano_net.pth`
- **Epochs**: 30 max (early stopping may trigger earlier)
- **Batch size**: 16 (set in `__main__`)
- **num_workers**: 4 (safe because dataset loads pre-computed .pt tensors, no librosa)

Auto-detects preprocessed data in priority order:
1. `preprocessed_v2/manifest.json` (current, 81k chunks)
2. `preprocessed/manifest.json` (old v1, superseded)
3. Raw audio fallback (slow, num_workers=0)

---

## 7. Inference (inference.py)

### Multi-window strategy
1. Load audio → mono → resample to 22050 Hz → trim silence (librosa)
2. Extract all overlapping 16384-sample windows (50% overlap)
3. For each window: compute mel spectrogram on GPU → run model → softmax
4. Average all softmax probability vectors
5. Argmax of average → predicted note

This makes inference robust to which part of the recording is analyzed (attack vs decay).

### Config constants
Must match `preprocess_v2.py` exactly:
```python
TARGET_SR   = 22050
WINDOW_SIZE = 16384
CHUNK_HOP   = 8192
N_FFT       = 2048
HOP_LENGTH  = 256
N_MELS      = 128
TOP_DB      = 30
```

---

## 8. GUI (gui.py)

Split-panel interface:
- **Left**: folder selector + scrollable file list (auto-populated with .wav/.ogg/.flac/.mp3/.aiff)
- **Right**: predicted note (large), confidence %, window count, runner-ups

**Interaction**: single click on a filename → immediately starts audio playback (sounddevice) AND inference (background thread) simultaneously. Results update when inference completes.

**Stale result protection**: version counter ensures only the most recently clicked file's result is displayed, even if an older inference finishes later.

**Audio playback**: `sounddevice` + `torchaudio` (handles all formats; resamples to 22050 Hz).

**Model**: loaded once at startup, stays in VRAM for the entire session.

---

## 9. Windows / Environment Notes

- **Virtual env**: `.venv` in project root — always use `.venv/Scripts/python.exe`
- **CUDA**: torch 2.5.1 + CUDA 12.1, verified working on this machine
- **num_workers > 0**: safe ONLY when DataLoader loads pre-computed .pt files (no librosa calls)
- **OGG encoder limit**: torchaudio crashes saving OGG files longer than ~25 seconds → cap at 20s
- **SIGPIPE from piping**: running `python script.py | head` will kill the script silently on Windows MSYS bash

---

## 10. Performance History

| Run | Dataset | Val Acc | Test Acc | Key changes |
|-----|---------|---------|----------|-------------|
| v1 raw | ~1,276 raw files | ~50.9% | — | Baseline, CPU, no preprocessing |
| v1 preprocessed | ~5,371 chunks (8k window) | 82.7% | 82.7% | LR scheduler, early stopping, batch=32 |
| v2 preprocessed | 81,147 chunks (16k window) | **98.2%** | **97.9%** | GPU pipeline, doubled window, 50% overlap, 15× more data |
