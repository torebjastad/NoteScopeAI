// Radix-2 Cooley-Tukey FFT for power-of-2 sizes
// Optimized for repeated calls with the same size (caches twiddle factors)

const _cache = new Map();

function _getTwiddles(n) {
    if (_cache.has(n)) return _cache.get(n);
    const cos = new Float64Array(n / 2);
    const sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
        const angle = -2 * Math.PI * i / n;
        cos[i] = Math.cos(angle);
        sin[i] = Math.sin(angle);
    }
    _cache.set(n, { cos, sin });
    return { cos, sin };
}

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * @param {Float64Array} re - Real part (length must be power of 2)
 * @param {Float64Array} im - Imaginary part (same length)
 */
export function fft(re, im) {
    const n = re.length;
    // Bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            [re[i], re[j]] = [re[j], re[i]];
            [im[i], im[j]] = [im[j], im[i]];
        }
    }

    const tw = _getTwiddles(n);
    // Butterfly stages
    for (let len = 2; len <= n; len *= 2) {
        const half = len / 2;
        const step = n / len;
        for (let i = 0; i < n; i += len) {
            for (let j = 0; j < half; j++) {
                const twIdx = j * step;
                const tRe = tw.cos[twIdx] * re[i + j + half] - tw.sin[twIdx] * im[i + j + half];
                const tIm = tw.cos[twIdx] * im[i + j + half] + tw.sin[twIdx] * re[i + j + half];
                re[i + j + half] = re[i + j] - tRe;
                im[i + j + half] = im[i + j] - tIm;
                re[i + j] += tRe;
                im[i + j] += tIm;
            }
        }
    }
}

/**
 * Compute magnitude squared of first n/2+1 FFT bins.
 * @param {Float64Array} re - Real part after FFT
 * @param {Float64Array} im - Imaginary part after FFT
 * @returns {Float32Array} Power spectrum of length n/2+1
 */
export function powerSpectrum(re, im) {
    const n = re.length;
    const nBins = n / 2 + 1;
    const power = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) {
        power[i] = re[i] * re[i] + im[i] * im[i];
    }
    return power;
}
