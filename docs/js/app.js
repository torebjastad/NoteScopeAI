// Main app controller — state machine, wires audio capture to inference to UI

import { init as initInference, isReady, infer } from './inference.js';
import { startCapture, stopCapture } from './audio-capture.js';
import { drawSpectrogram, clearSpectrogram } from './spectrogram-canvas.js';

// ── DOM refs ────────────────────────────────────────────────────────────────
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText    = document.getElementById('loading-text');
const noteDisplay    = document.getElementById('note-display');
const detectedNote   = document.getElementById('detected-note');
const barFill        = document.getElementById('bar-fill');
const confidenceText = document.getElementById('confidence-text');
const ru1Name        = document.getElementById('ru1-name');
const ru1Prob        = document.getElementById('ru1-prob');
const ru2Name        = document.getElementById('ru2-name');
const ru2Prob        = document.getElementById('ru2-prob');
const melCanvas      = document.getElementById('mel-canvas');
const startBtn       = document.getElementById('start-btn');
const statusText     = document.getElementById('status-text');
const latencyText    = document.getElementById('latency-text');

// ── State ───────────────────────────────────────────────────────────────────
let state = 'LOADING';  // LOADING | READY | LISTENING | MIC_DENIED
let _inferring = false;
let _prevNote = '';

// ── Initialize ──────────────────────────────────────────────────────────────
async function boot() {
    try {
        await initInference((msg) => {
            loadingText.textContent = msg;
        });
        loadingOverlay.classList.add('hidden');
        setState('READY');
    } catch (err) {
        loadingText.textContent = `Failed to load model: ${err.message}`;
        console.error('Init error:', err);
    }
}

function setState(newState) {
    state = newState;
    switch (state) {
        case 'READY':
            startBtn.disabled = false;
            startBtn.textContent = 'Start Listening';
            startBtn.classList.remove('listening');
            noteDisplay.classList.remove('listening');
            statusText.textContent = 'Tap to begin';
            statusText.classList.remove('error');
            latencyText.textContent = '';
            break;
        case 'LISTENING':
            startBtn.disabled = false;
            startBtn.textContent = 'Stop';
            startBtn.classList.add('listening');
            noteDisplay.classList.add('listening');
            statusText.textContent = 'Listening...';
            statusText.classList.remove('error');
            break;
        case 'MIC_DENIED':
            startBtn.disabled = false;
            startBtn.textContent = 'Retry';
            startBtn.classList.remove('listening');
            noteDisplay.classList.remove('listening');
            statusText.textContent = 'Microphone access denied. Please allow and retry.';
            statusText.classList.add('error');
            break;
    }
}

// ── Button handler ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
    if (state === 'LISTENING') {
        stopCapture();
        setState('READY');
        return;
    }

    try {
        statusText.textContent = 'Requesting microphone...';
        const nativeSR = await startCapture(onAudioBuffer);
        statusText.textContent = `Mic open (${nativeSR} Hz)`;
        setState('LISTENING');
    } catch (err) {
        console.error('Mic error:', err);
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            setState('MIC_DENIED');
        } else {
            statusText.textContent = `Mic error: ${err.message}`;
            statusText.classList.add('error');
        }
    }
});

// ── Audio buffer callback (called by audio-capture.js) ──────────────────────
async function onAudioBuffer(samples) {
    if (state !== 'LISTENING') return;
    if (_inferring) return;  // skip if previous inference still running

    _inferring = true;
    const t0 = performance.now();

    try {
        const result = await infer(samples);
        const latencyMs = performance.now() - t0;

        // Update UI
        updateNote(result.noteName);
        updateConfidence(result.confidence);
        updateRunnerUps(result.top3);
        drawSpectrogram(melCanvas, result.melDb, result.nFrames);
        latencyText.textContent = `${Math.round(latencyMs)} ms`;
    } catch (err) {
        console.error('Inference error:', err);
        statusText.textContent = `Error: ${err.message}`;
    } finally {
        _inferring = false;
    }
}

// ── UI update helpers ───────────────────────────────────────────────────────
function updateNote(name) {
    if (name !== _prevNote) {
        detectedNote.classList.add('updating');
        requestAnimationFrame(() => {
            detectedNote.textContent = name;
            requestAnimationFrame(() => {
                detectedNote.classList.remove('updating');
            });
        });
        _prevNote = name;
    }
}

function updateConfidence(conf) {
    const pct = (conf * 100).toFixed(1);
    barFill.style.width = `${pct}%`;
    confidenceText.textContent = `${pct}%`;
}

function updateRunnerUps(top3) {
    if (top3.length >= 2) {
        ru1Name.textContent = top3[1].name;
        ru1Prob.textContent = `${(top3[1].prob * 100).toFixed(1)}%`;
    }
    if (top3.length >= 3) {
        ru2Name.textContent = top3[2].name;
        ru2Prob.textContent = `${(top3[2].prob * 100).toFixed(1)}%`;
    }
}

// ── Init spectrogram canvas and boot ────────────────────────────────────────
clearSpectrogram(melCanvas);
boot();
