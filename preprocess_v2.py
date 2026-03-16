"""
preprocess_v2.py — Two-stage GPU-accelerated preprocessing pipeline.

Stage A: Load verified audio -> resample -> GPU trim -> GPU pitch shift -> save as OGG
  Output: processed_audio/{note_name}/shift_{tag}/{stem}.ogg
          processed_audio/manifest_a.json

Stage B: Load OGG -> extract overlapping 16384-sample chunks -> GPU mel spec -> save .pt
  Output: preprocessed_v2/specs/*.pt
          preprocessed_v2/manifest.json

Validation: Compare torch-trim (GPU) vs librosa-trim (CPU) on a small subset.

Usage:
  python preprocess_v2.py --validate            # validate on 20 files, then stop
  python preprocess_v2.py --stage a             # Stage A only
  python preprocess_v2.py --stage b             # Stage B only
  python preprocess_v2.py                       # validate, then Stage A + B
"""

import os, json, re, argparse, time
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
import librosa
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_SR      = 22050
WINDOW_SIZE    = 16384          # ~743 ms at 22050 Hz (doubled from v1)
CHUNK_HOP      = WINDOW_SIZE // 2   # 50% overlap = 8192 samples
N_FFT          = 2048           # better freq resolution for larger window
HOP_LENGTH     = 256            # mel spec hop -> 65 time frames per window
N_MELS         = 128
TOP_DB         = 30             # silence threshold
MIN_RMS_RATIO  = 0.02           # chunk must have RMS > 2% of file peak to be kept
PITCH_SHIFTS   = [-2, -1, 0, 1, 2]

VERIFIED_JSON    = "verified_files.json"
STAGE_A_DIR      = "processed_audio"
STAGE_A_MANIFEST = os.path.join(STAGE_A_DIR, "manifest_a.json")
STAGE_B_DIR      = "preprocessed_v2"
SPECS_DIR        = os.path.join(STAGE_B_DIR, "specs")
MANIFEST_PATH    = os.path.join(STAGE_B_DIR, "manifest.json")

AUDIO_EXT        = "ogg"        # fallback to "wav" if OGG save fails
# ──────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-built GPU transforms (reused across all files)
_mel_transform = AT.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
).to(device)
_amp_to_db = AT.AmplitudeToDB().to(device)


# ── Helpers ───────────────────────────────────────────────────────────────────

NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
}
SEMITONE_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def label_from_filepath(filepath: str) -> int:
    filename = os.path.basename(filepath)
    match = re.search(r"([A-G][b#]?\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse label from: {filename}")
    note_str = match.group(1)
    m = re.match(r"([A-G][b#]?)(-?\d+)", note_str)
    if not m:
        raise ValueError(f"Bad note: {note_str}")
    pitch_class, octave = m.groups()
    midi = (int(octave) + 1) * 12 + NOTE_TO_SEMITONE[pitch_class]
    idx = midi - 21
    if idx < 0 or idx > 87:
        raise ValueError(f"Out of piano range: {note_str}")
    return idx


def index_to_note_name(index: int) -> str:
    midi = index + 21
    return f"{SEMITONE_TO_NOTE[midi % 12]}{(midi // 12) - 1}"


def trim_silence_torch(waveform: torch.Tensor, top_db: float = 30.0,
                       frame_length: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    GPU-compatible silence trim — equivalent to librosa.effects.trim.
    waveform: [1, T] on any device. Returns [1, T_trimmed].
    """
    x = waveform[0]
    T = len(x)
    if T < frame_length:
        return waveform

    # Vectorised RMS per frame via unfold
    n_frames = 1 + (T - frame_length) // hop_length
    frames = x.unfold(0, frame_length, hop_length)[:n_frames]   # [n_frames, frame_length]
    rms = frames.pow(2).mean(-1).sqrt()                          # [n_frames]

    ref = rms.max()
    if ref == 0:
        return waveform

    threshold = ref * (10.0 ** (-top_db / 20.0))
    non_silent = (rms > threshold).nonzero(as_tuple=True)[0]

    if len(non_silent) == 0:
        return waveform

    start = int(non_silent[0].item() * hop_length)
    end   = min(int(non_silent[-1].item() * hop_length) + frame_length, T)
    return waveform[:, start:end]


def load_and_trim_gpu(filepath: str) -> torch.Tensor:
    """Load -> mono -> resample -> GPU -> trim. Returns [1, T] on GPU."""
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = AF.resample(waveform, sr, TARGET_SR)
    waveform = waveform.to(device)
    return trim_silence_torch(waveform, top_db=TOP_DB)


def extract_chunks(waveform: torch.Tensor) -> list:
    """
    Extract overlapping WINDOW_SIZE chunks (50% overlap).
    Filters chunks whose RMS < MIN_RMS_RATIO * file_peak_rms.
    Returns list of [1, WINDOW_SIZE] GPU tensors.
    """
    T = waveform.shape[1]
    if T == 0:
        return []

    peak_rms = waveform.pow(2).mean().sqrt().item()
    rms_thresh = peak_rms * MIN_RMS_RATIO

    chunks = []
    start = 0
    while start + WINDOW_SIZE <= T:
        chunk = waveform[:, start:start + WINDOW_SIZE]
        if chunk.pow(2).mean().sqrt().item() >= rms_thresh:
            chunks.append(chunk)
        start += CHUNK_HOP

    # If no full window fits, pad the whole trimmed audio
    if not chunks and T > 0:
        padded = torch.nn.functional.pad(waveform, (0, WINDOW_SIZE - T))
        chunks.append(padded)

    return chunks


def to_mel_spec(chunk: torch.Tensor) -> torch.Tensor:
    """chunk: [1, WINDOW_SIZE] GPU -> spec [1, N_MELS, T_frames] GPU"""
    return _amp_to_db(_mel_transform(chunk))


MAX_SAVE_SAMPLES = int(TARGET_SR * 20)   # OGG encoder crashes on >~25s; cap at 20s

def try_save_ogg(path: str, waveform_cpu: torch.Tensor) -> str:
    """Save as OGG (capped at MAX_SAVE_SAMPLES); fall back to WAV if OGG encoding is unavailable."""
    # Truncate very long files to avoid OGG encoder crash
    if waveform_cpu.shape[-1] > MAX_SAVE_SAMPLES:
        waveform_cpu = waveform_cpu[..., :MAX_SAVE_SAMPLES]
    try:
        torchaudio.save(path, waveform_cpu, TARGET_SR, format="ogg")
        return path
    except Exception:
        wav_path = path.replace(".ogg", ".wav")
        torchaudio.save(wav_path, waveform_cpu, TARGET_SR)
        return wav_path


# ── Validation ────────────────────────────────────────────────────────────────

def validate_gpu_vs_cpu(n_files: int = 20):
    """
    For n_files, compute mel spec on CPU (librosa trim) and GPU (torch trim)
    and compare. Reports mean/max absolute error in dB.
    Also benchmarks GPU vs CPU speed.
    """
    print(f"\n=== Validation: GPU vs CPU pipeline on {n_files} files ===")
    print(f"Device: {device}\n")

    with open(VERIFIED_JSON) as f:
        files = json.load(f)[:n_files]

    # CPU mel transform (same params)
    cpu_mel    = AT.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT,
                                   hop_length=HOP_LENGTH, n_mels=N_MELS)
    cpu_amp_db = AT.AmplitudeToDB()

    mae_list, onset_diff_ms = [], []
    t_cpu_total = t_gpu_total = 0.0

    for filepath in tqdm(files, desc="Validating"):
        try:
            # ── CPU path (librosa trim) ──────────────────────────────────────
            t0 = time.perf_counter()
            wf, sr = torchaudio.load(filepath)
            if wf.shape[0] > 1:
                wf = wf.mean(0, keepdim=True)
            if sr != TARGET_SR:
                wf = AF.resample(wf, sr, TARGET_SR)
            wav_np = wf.numpy()[0]
            trimmed_np, (onset_cpu, _) = librosa.effects.trim(wav_np, top_db=TOP_DB)
            wf_cpu = torch.from_numpy(trimmed_np).unsqueeze(0)
            if wf_cpu.shape[1] >= WINDOW_SIZE:
                w_cpu = wf_cpu[:, :WINDOW_SIZE]
            else:
                w_cpu = torch.nn.functional.pad(wf_cpu, (0, WINDOW_SIZE - wf_cpu.shape[1]))
            spec_cpu = cpu_amp_db(cpu_mel(w_cpu))
            t_cpu_total += time.perf_counter() - t0

            # ── GPU path (torch trim) ────────────────────────────────────────
            t0 = time.perf_counter()
            wf2, sr2 = torchaudio.load(filepath)
            if wf2.shape[0] > 1:
                wf2 = wf2.mean(0, keepdim=True)
            if sr2 != TARGET_SR:
                wf2 = AF.resample(wf2, sr2, TARGET_SR)
            wf2 = wf2.to(device)
            wf2_trim = trim_silence_torch(wf2, top_db=TOP_DB)
            # Approximate onset from GPU trim
            onset_gpu = wf2.shape[1] - wf2_trim.shape[1]   # samples removed from start
            if wf2_trim.shape[1] >= WINDOW_SIZE:
                w_gpu = wf2_trim[:, :WINDOW_SIZE]
            else:
                w_gpu = torch.nn.functional.pad(wf2_trim, (0, WINDOW_SIZE - wf2_trim.shape[1]))
            spec_gpu = _amp_to_db(_mel_transform(w_gpu)).cpu()
            torch.cuda.synchronize() if device.type == "cuda" else None
            t_gpu_total += time.perf_counter() - t0

            mae = (spec_cpu - spec_gpu).abs().mean().item()
            mae_list.append(mae)

            onset_diff_ms.append(abs(onset_cpu - onset_gpu) / TARGET_SR * 1000)

        except Exception as e:
            print(f"  Error on {os.path.basename(filepath)}: {e}")

    if not mae_list:
        print("No files processed.")
        return False

    THRESHOLD_DB = 2.0
    passed = sum(e < THRESHOLD_DB for e in mae_list)

    print(f"\nMel spectrogram MAE (CPU librosa-trim vs GPU torch-trim):")
    print(f"  Mean: {np.mean(mae_list):.4f} dB   Max: {np.max(mae_list):.4f} dB")
    print(f"  Within {THRESHOLD_DB} dB threshold: {passed}/{len(mae_list)} files")
    print(f"\nOnset detection difference:")
    print(f"  Mean: {np.mean(onset_diff_ms):.1f} ms   Max: {np.max(onset_diff_ms):.1f} ms")
    print(f"\nSpeed ({n_files} files, no caching):")
    print(f"  CPU total: {t_cpu_total:.2f}s  ({t_cpu_total/n_files:.3f}s/file)")
    print(f"  GPU total: {t_gpu_total:.2f}s  ({t_gpu_total/n_files:.3f}s/file)")
    if t_cpu_total > 0:
        print(f"  Speedup:   {t_cpu_total/t_gpu_total:.1f}x")

    ok = passed == len(mae_list)
    print(f"\n{'PASS' if ok else 'FAIL'} — GPU pipeline {'is' if ok else 'is NOT'} equivalent to CPU pipeline.")
    return ok


# ── Stage A: Pitch shift -> OGG ───────────────────────────────────────────────

def stage_a_pitch_shift():
    print(f"\n=== Stage A: Pitch shift -> {AUDIO_EXT.upper()} (device={device}) ===")
    os.makedirs(STAGE_A_DIR, exist_ok=True)

    with open(VERIFIED_JSON) as f:
        verified = json.load(f)

    manifest_a = []
    errors = 0
    skipped = 0
    t0 = time.perf_counter()

    for filepath in tqdm(verified, desc="Stage A"):
        try:
            label = label_from_filepath(filepath)
            stem  = os.path.splitext(os.path.basename(filepath))[0]

            waveform = None  # lazy-load: only needed if any shift is missing

            for shift in PITCH_SHIFTS:
                shifted_label = label + shift
                if shifted_label < 0 or shifted_label > 87:
                    continue

                note_name = index_to_note_name(shifted_label)
                shift_tag = f"p{shift}" if shift >= 0 else f"m{abs(shift)}"
                out_dir   = os.path.join(STAGE_A_DIR, note_name, f"shift_{shift_tag}")
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{stem}.{AUDIO_EXT}")
                alt_path = out_path.replace(".ogg", ".wav")

                # Resume: skip if already processed
                if os.path.exists(out_path):
                    manifest_a.append({"path": out_path, "label": shifted_label})
                    skipped += 1
                    continue
                if os.path.exists(alt_path):
                    manifest_a.append({"path": alt_path, "label": shifted_label})
                    skipped += 1
                    continue

                # Lazy-load waveform on first missing shift for this file
                if waveform is None:
                    waveform = load_and_trim_gpu(filepath)   # [1, T] on GPU

                if shift == 0:
                    shifted = waveform
                else:
                    shifted = AF.pitch_shift(waveform, TARGET_SR, float(shift))

                saved_path = try_save_ogg(out_path, shifted.cpu())
                manifest_a.append({"path": saved_path, "label": shifted_label})

        except Exception as e:
            print(f"\n  Error on {os.path.basename(filepath)}: {e}")
            errors += 1

    elapsed = time.perf_counter() - t0
    with open(STAGE_A_MANIFEST, "w") as f:
        json.dump(manifest_a, f, indent=2)

    new_files = len(manifest_a) - skipped
    print(f"\nStage A done in {elapsed/60:.1f} min.")
    print(f"  {len(manifest_a)} total entries ({skipped} skipped/resumed, {new_files} new), {errors} errors.")
    print(f"  Manifest: {STAGE_A_MANIFEST}")


# ── Stage B: Chunks -> mel spectrograms ───────────────────────────────────────

def stage_b_mel_spectrograms():
    print(f"\n=== Stage B: Chunks -> mel spectrograms (device={device}) ===")

    if not os.path.exists(STAGE_A_MANIFEST):
        print(f"ERROR: {STAGE_A_MANIFEST} not found. Run Stage A first.")
        return

    os.makedirs(SPECS_DIR, exist_ok=True)

    with open(STAGE_A_MANIFEST) as f:
        manifest_a = json.load(f)

    print(f"Processing {len(manifest_a)} audio files -> chunks -> .pt")
    print(f"Window: {WINDOW_SIZE} samples ({WINDOW_SIZE/TARGET_SR*1000:.0f} ms), "
          f"hop: {CHUNK_HOP} samples (50% overlap)")

    manifest_b = []
    errors = 0
    total_chunks = 0
    file_idx = 0
    t0 = time.perf_counter()

    for entry in tqdm(manifest_a, desc="Stage B"):
        audio_path = entry["path"]
        label      = entry["label"]

        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            if sr != TARGET_SR:
                waveform = AF.resample(waveform, sr, TARGET_SR)
            waveform = waveform.to(device)

            chunks = extract_chunks(waveform)

            for chunk_idx, chunk in enumerate(chunks):
                spec = to_mel_spec(chunk)   # [1, N_MELS, T_frames] GPU
                fname = f"{file_idx:05d}_lbl{label:02d}_c{chunk_idx:02d}.pt"
                out_path = os.path.join(SPECS_DIR, fname)
                torch.save(spec.cpu(), out_path)
                manifest_b.append({"path": out_path, "label": label})
                total_chunks += 1

            file_idx += 1

        except Exception as e:
            print(f"\n  Error on {audio_path}: {e}")
            errors += 1

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_b, f)

    elapsed = time.perf_counter() - t0
    avg_chunks = total_chunks / max(file_idx, 1)
    print(f"\nStage B done in {elapsed/60:.1f} min.")
    print(f"  {total_chunks} spectrogram files written ({avg_chunks:.1f} chunks/file avg), {errors} errors.")
    print(f"  Manifest: {MANIFEST_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated preprocessing pipeline v2")
    parser.add_argument("--validate", action="store_true",
                        help="Validate GPU vs CPU on a subset before running")
    parser.add_argument("--n-validate", type=int, default=20,
                        help="Number of files for validation (default: 20)")
    parser.add_argument("--stage", type=str, default="ab", choices=["a", "b", "ab"],
                        help="Which stage(s) to run (default: ab)")
    parser.add_argument("--skip-validate", action="store_true",
                        help="Skip validation even when running full pipeline")
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"Window: {WINDOW_SIZE} samples | Hop: {CHUNK_HOP} | N_FFT: {N_FFT} | SR: {TARGET_SR}")

    # Always validate first unless explicitly skipped
    if not args.skip_validate:
        ok = validate_gpu_vs_cpu(n_files=args.n_validate)
        if args.validate:
            return      # --validate flag: stop after validation
        if not ok:
            print("\nValidation failed. Fix issues before running full pipeline.")
            print("Use --skip-validate to bypass (not recommended).")
            return
        print("\nValidation passed. Starting preprocessing...\n")

    if "a" in args.stage:
        stage_a_pitch_shift()

    if "b" in args.stage:
        stage_b_mel_spectrograms()

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
