// Mel spectrogram computation matching torchaudio.transforms.MelSpectrogram + AmplitudeToDB
// Config must match inference.py constants exactly.

import { fft, powerSpectrum } from './fft.js';

// ── Config (matches inference.py) ───────────────────────────────────────────
export const TARGET_SR    = 22050;
export const WINDOW_SIZE  = 16384;
export const N_FFT        = 2048;
export const HOP_LENGTH   = 256;
export const N_MELS       = 128;
const N_BINS              = N_FFT / 2 + 1;  // 1025
const TOP_DB              = 80;              // AmplitudeToDB default
const AMIN                = 1e-10;           // AmplitudeToDB default
// ────────────────────────────────────────────────────────────────────────────

// Pre-computed Hann window
const _hannWindow = new Float64Array(N_FFT);
for (let i = 0; i < N_FFT; i++) {
    _hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / N_FFT));
}

let _filterbank = null;  // [N_MELS x N_BINS] Float32Array rows

/**
 * Load mel filterbank from JSON file.
 * Must be called before computeMelSpectrogram.
 */
export async function loadFilterbank(url = './model/mel_filterbank.json') {
    const resp = await fetch(url);
    const data = await resp.json();
    _filterbank = data.filterbank;  // array of 128 arrays, each length 1025
}

/**
 * Reflect-pad a signal by `pad` samples on each side.
 * Matches PyTorch's pad_mode="reflect" for MelSpectrogram(center=True).
 */
function reflectPad(signal, pad) {
    const n = signal.length;
    const out = new Float32Array(n + 2 * pad);
    // Left padding: reflect
    for (let i = 0; i < pad; i++) {
        out[pad - 1 - i] = signal[i + 1];
    }
    // Center: copy signal
    out.set(signal, pad);
    // Right padding: reflect
    for (let i = 0; i < pad; i++) {
        out[n + pad + i] = signal[n - 2 - i];
    }
    return out;
}

/**
 * Compute mel spectrogram for a single window of audio samples.
 * Input: Float32Array of WINDOW_SIZE (16384) samples at TARGET_SR (22050 Hz).
 * Output: { melDb: Float32Array(N_MELS * nFrames), nFrames: number }
 *   melDb is row-major [N_MELS, nFrames] in dB scale.
 */
export function computeMelSpectrogram(samples) {
    if (!_filterbank) throw new Error('Filterbank not loaded. Call loadFilterbank() first.');

    // Step 1: Reflect-pad (center=True, pad = n_fft // 2 = 1024)
    const padded = reflectPad(samples, N_FFT / 2);

    // Step 2: STFT — extract frames, apply Hann window, FFT
    const nFrames = Math.floor((padded.length - N_FFT) / HOP_LENGTH) + 1;  // should be 65
    const powerFrames = new Array(nFrames);

    const re = new Float64Array(N_FFT);
    const im = new Float64Array(N_FFT);

    for (let t = 0; t < nFrames; t++) {
        const offset = t * HOP_LENGTH;
        // Apply Hann window and copy to FFT buffers
        for (let i = 0; i < N_FFT; i++) {
            re[i] = padded[offset + i] * _hannWindow[i];
            im[i] = 0;
        }
        fft(re, im);
        powerFrames[t] = powerSpectrum(re, im);  // Float32Array[1025]
    }

    // Step 3: Apply mel filterbank
    // filterbank[m] is length 1025, powerFrames[t] is length 1025
    // melSpec[m, t] = sum_f(filterbank[m][f] * powerFrames[t][f])
    const melSpec = new Float32Array(N_MELS * nFrames);

    for (let m = 0; m < N_MELS; m++) {
        const fb = _filterbank[m];
        for (let t = 0; t < nFrames; t++) {
            const pf = powerFrames[t];
            let sum = 0;
            for (let f = 0; f < N_BINS; f++) {
                sum += fb[f] * pf[f];
            }
            melSpec[m * nFrames + t] = sum;
        }
    }

    // Step 4: Amplitude to dB
    // 10 * log10(max(S, amin)), then clamp to max - top_db
    let maxDb = -Infinity;
    for (let i = 0; i < melSpec.length; i++) {
        melSpec[i] = 10 * Math.log10(Math.max(melSpec[i], AMIN));
        if (melSpec[i] > maxDb) maxDb = melSpec[i];
    }
    const clampMin = maxDb - TOP_DB;
    for (let i = 0; i < melSpec.length; i++) {
        if (melSpec[i] < clampMin) melSpec[i] = clampMin;
    }

    return { melDb: melSpec, nFrames };
}
