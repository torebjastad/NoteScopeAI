# Agent Instructions

Guidelines for working on this project. Read knowledge.md for full technical context.

---

## Project at a Glance

Piano note classifier: 88-class CNN, audio → mel spectrogram → predicted note.
- **Model**: `piano_net.pth` (98.2% val accuracy)
- **Virtual env**: always use `.venv/Scripts/python.exe` — not system Python
- **GPU**: CUDA 12.1 available and used for preprocessing and training

---

## File Map

```
NoteScopeAI/
├── model.py              — PianoNet CNN (3 conv blocks + AdaptiveAvgPool + FC)
├── train.py              — Training loop; auto-detects preprocessed_v2 manifest
├── dataset.py            — PreprocessedPianoDataset (.pt files) + PianoToneDataset (raw)
├── inference.py          — Multi-window inference: average softmax over all windows
├── gui.py                — Tkinter GUI: folder browser, click-to-play + infer
├── preprocess_v2.py      — GPU two-stage pipeline (Stage A: pitch shift→OGG, Stage B: chunks→.pt)
├── preprocess.py         — Old v1 CPU pipeline (superseded, keep for reference)
├── verify_dataset.py     — Pitch verification using UltimatePianoDetector
│
├── verified_files.json   — 1,088 verified source audio file paths (training allowlist)
├── failed_files_report.txt — 188 excluded files (useful as unseen test samples)
│
├── piano_net.pth         — Current best model (for inference/GUI)
├── piano_net_best.pth    — Best checkpoint from last training run
├── piano_net_test.pth    — Snapshot copy for safe testing while retraining
│
├── processed_audio/      — Stage A output: OGG files per note/shift
│   ├── manifest_a.json   — 5,371 entries: {path, label}
│   └── {note}/shift_{tag}/{stem}.ogg
│
├── preprocessed_v2/      — Stage B output: mel spectrogram tensors
│   ├── manifest.json     — 81,147 entries: {path, label}
│   └── specs/{idx}_lbl{label}_c{chunk}.pt  ← shape [1, 128, ~65]
│
├── knowledge.md          — Technical knowledge base (READ THIS FIRST)
├── agent_instructions.md — This file
├── setup_env.ps1         — One-time env setup script
└── .venv/                — Python 3.12 virtual environment
```

---

## Critical Rules

### Always use .venv
```bash
.venv/Scripts/python.exe script.py
```
Never use system Python — the torch/torchaudio/sounddevice packages are only in .venv.

### Never pipe script output to head/grep
tqdm uses `\r` carriage returns. Piping to `head` or `grep` causes SIGPIPE which silently
kills the Python process. Use log file redirection instead:
```bash
.venv/Scripts/python.exe script.py > output.log 2>&1
```

### num_workers
- `num_workers=4` is safe when DataLoader loads pre-computed `.pt` files (PreprocessedPianoDataset)
- `num_workers=0` required if loading raw audio with librosa (PianoToneDataset)

### OGG encoder length limit
`torchaudio.save(..., format="ogg")` crashes on audio longer than ~25 seconds.
The 20-second cap in `try_save_ogg()` in `preprocess_v2.py` handles this — don't remove it.

### Model loading
Always use `weights_only=True` when calling `torch.load()`:
```python
torch.load("piano_net.pth", map_location=device, weights_only=True)
```

---

## Common Tasks

### Run training from scratch
```bash
.venv/Scripts/python.exe train.py
```
Automatically uses `preprocessed_v2/manifest.json` if present. Saves best model to
`piano_net_best.pth` during training, then copies to `piano_net.pth` at the end.

### Re-preprocess after adding new audio files
1. Add new files to the Piano source folder
2. Run verification: `.venv/Scripts/python.exe verify_dataset.py`
3. Review `failed_files_report.txt`; update `verified_files.json` as needed
4. Stage A: `.venv/Scripts/python.exe preprocess_v2.py --stage a --skip-validate`
   - Has resume support — safe to interrupt and restart
5. Stage B: `.venv/Scripts/python.exe preprocess_v2.py --stage b --skip-validate`
6. Re-run training

### Run inference on a single file (CLI)
```bash
.venv/Scripts/python.exe inference.py path/to/audio.wav
```

### Run GUI
```bash
.venv/Scripts/python.exe gui.py
```

---

## Architecture Constraints — Don't Change These

- **Config constants** in `inference.py` and `preprocess_v2.py` must stay in sync:
  `TARGET_SR=22050, WINDOW_SIZE=16384, CHUNK_HOP=8192, N_FFT=2048, HOP_LENGTH=256, N_MELS=128`
- **Pitch shift range**: ±2 semitones maximum. Beyond ±2, piano body resonances distort.
- **AdaptiveAvgPool2d(8,4)** in model.py is what makes the model accept variable-length inputs.
  Changing the pool output size also changes the FC layer sizes (128*8*4=4096).

---

## Data Flow Summary

```
Raw audio (.wav/.ogg)
    ↓ verify_dataset.py (pitch check, 40–100 cent tolerance, octave errors OK)
verified_files.json (1,088 files)
    ↓ preprocess_v2.py --stage a
processed_audio/ (5,371 OGG files, ±2 semitone variants)
    ↓ preprocess_v2.py --stage b
preprocessed_v2/ (81,147 .pt tensors, shape [1, 128, ~65])
    ↓ train.py
piano_net.pth (trained model, 98.2% accuracy)
    ↓ inference.py / gui.py
predicted note + confidence
```
