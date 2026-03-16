"""Export PianoNet to ONNX format and extract mel filterbank weights for browser inference."""

import json
import os
import numpy as np
import torch
import torchaudio.transforms as AT
import onnx
import onnxruntime as ort
from model import PianoNet

# ── Config (must match inference.py) ─────────────────────────────────────────
TARGET_SR  = 22050
N_FFT      = 2048
HOP_LENGTH = 256
N_MELS     = 128
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH   = "piano_net.pth"
ONNX_DIR     = os.path.join("docs", "model")
ONNX_PATH    = os.path.join(ONNX_DIR, "piano_net.onnx")
FB_PATH      = os.path.join(ONNX_DIR, "mel_filterbank.json")


def export_onnx():
    os.makedirs(ONNX_DIR, exist_ok=True)

    # Load model
    device = torch.device("cpu")
    model = PianoNet(num_classes=88)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Dummy input: [batch, channels, n_mels, time_frames]
    dummy = torch.randn(1, 1, 128, 65)

    # Export to ONNX
    print(f"Exporting model to {ONNX_PATH} ...")
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
    )

    # Validate ONNX model
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    onnx_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"ONNX model valid. Size: {onnx_size:.1f} MB")

    # Verify outputs match
    print("Verifying ONNX vs PyTorch outputs ...")
    sess = ort.InferenceSession(ONNX_PATH)
    max_diff = 0.0
    for i in range(5):
        test_input = torch.randn(1, 1, 128, 65)
        with torch.no_grad():
            pt_out = model(test_input).numpy()
        ort_out = sess.run(None, {"input": test_input.numpy()})[0]
        diff = np.max(np.abs(pt_out - ort_out))
        max_diff = max(max_diff, diff)
        print(f"  Test {i+1}: max abs diff = {diff:.2e}")
    print(f"  Max diff across all tests: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Output mismatch too large: {max_diff}"

    # Extract mel filterbank
    print(f"Extracting mel filterbank to {FB_PATH} ...")
    mel_t = AT.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    # fb shape in torchaudio: [n_freqs, n_mels] = [1025, 128]
    fb = mel_t.mel_scale.fb.numpy()  # [1025, 128]
    fb_transposed = fb.T  # [128, 1025] — each row is one mel filter
    fb_list = fb_transposed.tolist()

    with open(FB_PATH, "w") as f:
        json.dump({"filterbank": fb_list, "n_mels": N_MELS, "n_fft": N_FFT}, f)
    fb_size = os.path.getsize(FB_PATH) / 1024
    print(f"Filterbank saved. Size: {fb_size:.0f} KB")

    print("\nDone! Files ready for web deployment:")
    print(f"  {ONNX_PATH}")
    print(f"  {FB_PATH}")


if __name__ == "__main__":
    export_onnx()
