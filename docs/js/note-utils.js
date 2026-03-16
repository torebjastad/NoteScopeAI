// Note name utilities — port of inference.py:index_to_note_name

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

/**
 * Convert a model output index (0-87) to a note name like "A0", "C4", "C#5".
 * Index 0 = A0 (MIDI 21), Index 87 = C8 (MIDI 108).
 */
export function indexToNoteName(index) {
    if (index < 0 || index > 87) throw new RangeError(`Index out of bounds: ${index}`);
    const midi = index + 21;
    return `${NOTE_NAMES[midi % 12]}${Math.floor(midi / 12) - 1}`;
}

/**
 * Apply softmax to a Float32Array of logits. Returns a new Float32Array of probabilities.
 */
export function softmax(logits) {
    const max = logits.reduce((a, b) => Math.max(a, b), -Infinity);
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - max);
        sum += exps[i];
    }
    for (let i = 0; i < logits.length; i++) {
        exps[i] /= sum;
    }
    return exps;
}

/**
 * Get top-k predictions from probabilities array.
 * Returns array of { index, name, prob } sorted by probability descending.
 */
export function topK(probs, k = 3) {
    const indexed = Array.from(probs).map((p, i) => ({ index: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k).map(({ index, prob }) => ({
        index,
        name: indexToNoteName(index),
        prob,
    }));
}
