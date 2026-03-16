// ONNX Runtime Web inference wrapper for PianoNet

import { computeMelSpectrogram, loadFilterbank, N_MELS } from './mel-spectrogram.js';
import { softmax, topK } from './note-utils.js';

let _session = null;
let _ready = false;

/**
 * Initialize the inference engine: load filterbank + ONNX model + warm up.
 * @param {function} onProgress - Optional callback(message) for loading status.
 */
export async function init(onProgress) {
    // Ensure WASM files are loaded from CDN (not relative to page)
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

    onProgress?.('Loading mel filterbank...');
    await loadFilterbank('./model/mel_filterbank.json');

    onProgress?.('Loading neural network...');
    _session = await ort.InferenceSession.create('./model/piano_net.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
    });

    // Warm up with dummy input to avoid cold-start latency
    onProgress?.('Warming up...');
    const dummy = new Float32Array(1 * 1 * N_MELS * 65);
    const tensor = new ort.Tensor('float32', dummy, [1, 1, N_MELS, 65]);
    await _session.run({ input: tensor });

    _ready = true;
    onProgress?.('Ready');
}

export function isReady() {
    return _ready;
}

/**
 * Run inference on a window of 16384 audio samples at 22050 Hz.
 * Returns { noteName, confidence, top3, melDb, nFrames }
 */
export async function infer(samples) {
    if (!_ready) throw new Error('Inference engine not initialized');

    // Compute mel spectrogram
    const { melDb, nFrames } = computeMelSpectrogram(samples);

    // Create ONNX tensor [1, 1, 128, nFrames]
    const tensor = new ort.Tensor('float32', melDb, [1, 1, N_MELS, nFrames]);

    // Run model
    const results = await _session.run({ input: tensor });
    const logits = results.output.data;  // Float32Array[88]

    // Softmax + top-k
    const probs = softmax(logits);
    const top3 = topK(probs, 3);

    return {
        noteName: top3[0].name,
        confidence: top3[0].prob,
        top3,
        melDb,
        nFrames,
    };
}
