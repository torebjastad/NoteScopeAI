// AudioWorkletProcessor that accumulates mic samples and posts buffers to main thread.
// Runs in the audio rendering thread for low latency.

class CaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = options.processorOptions || {};
        this._nativeSR = opts.nativeSR || 44100;
        this._targetSR = opts.targetSR || 22050;
        this._windowSize = opts.windowSize || 16384;

        // How many native-rate samples we need to fill one target-rate window
        this._nativeSamplesNeeded = Math.ceil(this._windowSize * this._nativeSR / this._targetSR);
        this._buffer = new Float32Array(this._nativeSamplesNeeded);
        this._writePos = 0;
        this._active = true;

        this.port.onmessage = (e) => {
            if (e.data === 'stop') this._active = false;
            if (e.data === 'start') this._active = true;
        };
    }

    process(inputs) {
        if (!this._active) return true;

        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];  // mono channel, 128 samples per call

        for (let i = 0; i < channelData.length; i++) {
            this._buffer[this._writePos++] = channelData[i];

            if (this._writePos >= this._nativeSamplesNeeded) {
                // Buffer full — send to main thread
                this.port.postMessage({
                    type: 'buffer',
                    data: this._buffer.slice(0),  // copy
                    nativeSR: this._nativeSR,
                });
                this._writePos = 0;
            }
        }

        return true;  // keep processor alive
    }
}

registerProcessor('capture-processor', CaptureProcessor);
