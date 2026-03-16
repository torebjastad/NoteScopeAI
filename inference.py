import argparse
import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
import librosa
import numpy as np
from model import PianoNet

# ── Config (must match preprocess_v2.py) ──────────────────────────────────────
TARGET_SR   = 22050
WINDOW_SIZE = 16384
CHUNK_HOP   = WINDOW_SIZE // 2
N_FFT       = 2048
HOP_LENGTH  = 256
N_MELS      = 128
TOP_DB      = 30
# ──────────────────────────────────────────────────────────────────────────────

_mel_transform = None
_amp_to_db     = None

def _get_transforms(device):
    global _mel_transform, _amp_to_db
    if _mel_transform is None:
        _mel_transform = AT.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
        ).to(device)
        _amp_to_db = AT.AmplitudeToDB().to(device)
    return _mel_transform, _amp_to_db


def index_to_note_name(index: int) -> str:
    if index < 0 or index > 87:
        raise ValueError("Index out of bounds")
    midi = index + 21
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{notes[midi % 12]}{(midi // 12) - 1}"


def load_and_prepare(filepath: str) -> torch.Tensor:
    """Load audio, mix to mono, resample to TARGET_SR, trim silence. Returns CPU tensor [1, T]."""
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = AF.resample(waveform, sr, TARGET_SR)
    # Trim silence with librosa (CPU; fine for inference — not in training loop)
    wav_np = waveform.numpy()[0]
    trimmed_np, _ = librosa.effects.trim(wav_np, top_db=TOP_DB)
    return torch.from_numpy(trimmed_np).unsqueeze(0)


def extract_windows(waveform: torch.Tensor) -> list:
    """Extract overlapping WINDOW_SIZE windows with 50% overlap. Returns list of [1, WINDOW_SIZE]."""
    T = waveform.shape[1]
    windows = []
    start = 0
    while start + WINDOW_SIZE <= T:
        windows.append(waveform[:, start:start + WINDOW_SIZE])
        start += CHUNK_HOP
    if not windows:
        padded = torch.nn.functional.pad(waveform, (0, WINDOW_SIZE - T))
        windows.append(padded)
    return windows


def predict_note_multi_window(filepath: str, model, device) -> dict:
    """
    Load an audio file, run the model on every overlapping window,
    average the softmax probabilities, and return the aggregated prediction.

    Returns dict with keys:
      note_name, confidence, top3_names, top3_probs, n_windows,
      window_labels (list of int), window_confs (list of float)
    """
    mel_t, amp_db = _get_transforms(device)

    waveform = load_and_prepare(filepath)
    windows  = extract_windows(waveform)

    model.eval()
    avg_probs      = None
    window_labels  = []
    window_confs   = []

    with torch.no_grad():
        for w in windows:
            spec  = amp_db(mel_t(w.to(device))).unsqueeze(0)  # [1,1,N_MELS,T]
            probs = torch.softmax(model(spec), dim=1)
            conf, idx = torch.max(probs, 1)
            window_labels.append(idx.item())
            window_confs.append(conf.item())
            avg_probs = probs if avg_probs is None else avg_probs + probs

    avg_probs /= len(windows)

    confidence, pred_idx = torch.max(avg_probs, 1)
    top3_probs, top3_idx = torch.topk(avg_probs, 3, dim=1)

    return {
        "note_name":     index_to_note_name(pred_idx.item()),
        "confidence":    confidence.item(),
        "top3_names":    [index_to_note_name(i.item()) for i in top3_idx[0]],
        "top3_probs":    [p.item() for p in top3_probs[0]],
        "n_windows":     len(windows),
        "window_labels": window_labels,
        "window_confs":  window_confs,
    }


def predict_note(audio_path: str, model_path: str = "piano_net.pth") -> str:
    """CLI-friendly single-call wrapper — loads model and calls multi-window inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PianoNet(num_classes=88)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    result = predict_note_multi_window(audio_path, model, device)

    print(f"\n--- Prediction Results for {audio_path} ---")
    print(f"Predicted Note: {result['note_name']}")
    print(f"Confidence:     {result['confidence']:.2%}  (averaged over {result['n_windows']} window(s))")
    print("-" * 43)
    print("Top 3 Candidates:")
    for i, (name, prob) in enumerate(zip(result["top3_names"], result["top3_probs"])):
        print(f"  {i+1}. {name:<5} ({prob:.2%})")

    return result["note_name"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a piano note from an audio file.")
    parser.add_argument("audio_path", help="Path to .wav/.ogg audio file")
    parser.add_argument("--model", default="piano_net.pth", help="Path to trained model")
    args = parser.parse_args()
    predict_note(args.audio_path, args.model)
