// Canvas-based mel spectrogram visualization with magma colormap

import { N_MELS } from './mel-spectrogram.js';

// Magma colormap — 256-entry RGB lookup table
// Generated from matplotlib's magma colormap (key points interpolated)
const MAGMA_STOPS = [
    [0, 0, 4], [1, 1, 10], [3, 3, 21], [7, 4, 37], [14, 6, 55],
    [24, 8, 70], [35, 10, 82], [47, 11, 90], [60, 12, 95], [72, 12, 97],
    [85, 13, 97], [98, 16, 95], [111, 20, 91], [124, 25, 86], [136, 31, 80],
    [148, 38, 74], [158, 46, 68], [168, 54, 63], [177, 63, 58], [186, 72, 54],
    [194, 81, 50], [201, 91, 47], [209, 101, 44], [215, 112, 42], [221, 122, 40],
    [226, 133, 39], [231, 144, 38], [235, 156, 39], [238, 168, 42], [241, 180, 47],
    [243, 193, 55], [244, 205, 65], [245, 218, 79], [246, 231, 97], [249, 243, 118],
    [252, 253, 142],
];

const _colormap = new Uint8Array(256 * 3);
(function buildColormap() {
    const nStops = MAGMA_STOPS.length;
    for (let i = 0; i < 256; i++) {
        const t = i / 255 * (nStops - 1);
        const lo = Math.floor(t);
        const hi = Math.min(lo + 1, nStops - 1);
        const frac = t - lo;
        _colormap[i * 3 + 0] = Math.round(MAGMA_STOPS[lo][0] + frac * (MAGMA_STOPS[hi][0] - MAGMA_STOPS[lo][0]));
        _colormap[i * 3 + 1] = Math.round(MAGMA_STOPS[lo][1] + frac * (MAGMA_STOPS[hi][1] - MAGMA_STOPS[lo][1]));
        _colormap[i * 3 + 2] = Math.round(MAGMA_STOPS[lo][2] + frac * (MAGMA_STOPS[hi][2] - MAGMA_STOPS[lo][2]));
    }
})();

/**
 * Draw a mel spectrogram on a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {Float32Array} melDb - Row-major [N_MELS, nFrames] in dB
 * @param {number} nFrames
 */
export function drawSpectrogram(canvas, melDb, nFrames) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Find min/max for normalization
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < melDb.length; i++) {
        if (melDb[i] < min) min = melDb[i];
        if (melDb[i] > max) max = melDb[i];
    }
    const range = max - min || 1;

    // Create small ImageData at spectrogram resolution
    const imgData = ctx.createImageData(nFrames, N_MELS);
    const pixels = imgData.data;

    for (let m = 0; m < N_MELS; m++) {
        // Flip vertically: low frequencies at bottom (row 0 in data → bottom of image)
        const imgRow = N_MELS - 1 - m;
        for (let t = 0; t < nFrames; t++) {
            const val = melDb[m * nFrames + t];
            const norm = Math.max(0, Math.min(255, Math.round(255 * (val - min) / range)));
            const pixIdx = (imgRow * nFrames + t) * 4;
            pixels[pixIdx + 0] = _colormap[norm * 3 + 0];
            pixels[pixIdx + 1] = _colormap[norm * 3 + 1];
            pixels[pixIdx + 2] = _colormap[norm * 3 + 2];
            pixels[pixIdx + 3] = 255;
        }
    }

    // Draw small image to temp canvas, then scale up
    // Use regular canvas for broader compatibility (OffscreenCanvas not on all Safari)
    let tempCanvas = document.createElement('canvas');
    tempCanvas.width = nFrames;
    tempCanvas.height = N_MELS;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imgData, 0, 0);

    // Scale to canvas size with smooth interpolation
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tempCanvas, 0, 0, width, height);
}

/**
 * Clear the canvas with a dark placeholder.
 */
export function clearSpectrogram(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgba(201, 168, 76, 0.3)';
    ctx.font = '14px system-ui';
    ctx.textAlign = 'center';
    ctx.fillText('Mel spectrogram will appear here', canvas.width / 2, canvas.height / 2);
}
