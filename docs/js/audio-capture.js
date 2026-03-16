// Audio capture: microphone access, AudioWorklet setup, resampling to 22050 Hz

import { WINDOW_SIZE, TARGET_SR } from './mel-spectrogram.js';

let _audioContext = null;
let _workletNode = null;
let _stream = null;
let _onBuffer = null;  // callback(Float32Array of WINDOW_SIZE samples at TARGET_SR)

/**
 * Start microphone capture.
 * @param {function} onBuffer - Called with Float32Array(16384) at 22050 Hz each time a window is ready.
 * @returns {Promise<number>} Native sample rate.
 */
export async function startCapture(onBuffer) {
    _onBuffer = onBuffer;

    // Request microphone with raw audio (no processing)
    _stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
        },
    });

    _audioContext = new AudioContext();
    const nativeSR = _audioContext.sampleRate;

    // Load and connect AudioWorklet
    await _audioContext.audioWorklet.addModule('./worklet/capture-processor.js');
    _workletNode = new AudioWorkletNode(_audioContext, 'capture-processor', {
        processorOptions: {
            nativeSR,
            targetSR: TARGET_SR,
            windowSize: WINDOW_SIZE,
        },
    });

    _workletNode.port.onmessage = (e) => {
        if (e.data.type === 'buffer') {
            _resampleAndDeliver(e.data.data, e.data.nativeSR);
        }
    };

    const source = _audioContext.createMediaStreamSource(_stream);
    source.connect(_workletNode);
    // Don't connect worklet to destination (we don't want playback)

    return nativeSR;
}

/**
 * Resample native-rate buffer to TARGET_SR using OfflineAudioContext.
 */
async function _resampleAndDeliver(nativeBuffer, nativeSR) {
    if (!_onBuffer) return;

    if (nativeSR === TARGET_SR) {
        // No resampling needed — just trim/pad to WINDOW_SIZE
        const out = new Float32Array(WINDOW_SIZE);
        out.set(nativeBuffer.subarray(0, Math.min(nativeBuffer.length, WINDOW_SIZE)));
        _onBuffer(out);
        return;
    }

    // Use OfflineAudioContext for high-quality resampling
    const outLength = Math.round(nativeBuffer.length * TARGET_SR / nativeSR);
    const offline = new OfflineAudioContext(1, outLength, TARGET_SR);
    const buffer = offline.createBuffer(1, nativeBuffer.length, nativeSR);
    buffer.copyToChannel(nativeBuffer, 0);

    const source = offline.createBufferSource();
    source.buffer = buffer;
    source.connect(offline.destination);
    source.start();

    const rendered = await offline.startRendering();
    const resampled = rendered.getChannelData(0);

    // Ensure exactly WINDOW_SIZE samples
    const out = new Float32Array(WINDOW_SIZE);
    out.set(resampled.subarray(0, Math.min(resampled.length, WINDOW_SIZE)));
    _onBuffer(out);
}

/**
 * Pause audio capture (keeps mic open but stops sending buffers).
 */
export function pauseCapture() {
    _workletNode?.port.postMessage('stop');
}

/**
 * Resume audio capture after pause.
 */
export function resumeCapture() {
    _workletNode?.port.postMessage('start');
}

/**
 * Stop capture and release microphone.
 */
export function stopCapture() {
    _workletNode?.port.postMessage('stop');
    _workletNode?.disconnect();
    _workletNode = null;

    if (_stream) {
        _stream.getTracks().forEach(t => t.stop());
        _stream = null;
    }

    if (_audioContext) {
        _audioContext.close();
        _audioContext = null;
    }

    _onBuffer = null;
}
