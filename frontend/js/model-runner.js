/**
 * model-runner.js — ONNX Runtime Web inference for the shipped CNN.
 *
 * models/model.onnx was exported by train.py:_export_seq_onnx (train.py:374),
 * which deliberately leaves normalisation OUT of the graph — the caller must
 * apply (x - mean) / std first, exactly as predict.py:56-60 does. The scaler
 * lives in assets/seq_scaler.json (converted from models/seq_scaler.npz).
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export class ModelRunner {
  constructor(session, scaler) {
    this.session = session;
    this.scaler = scaler;
  }

  static async create(modelPath = 'assets/model.onnx', scalerPath = 'assets/seq_scaler.json') {
    // Fetch the bytes ourselves with the cache explicitly disabled, then hand
    // ort.InferenceSession.create() the buffer rather than a URL. Passing a URL
    // lets ort do its own fetch internally, which is not guaranteed to bypass a
    // stale HTTP cache entry on a normal reload — this bit us once already
    // (a previous export's weights got served from cache after the file on
    // disk had already changed).
    const [modelBytes, scaler] = await Promise.all([
      fetch(modelPath, { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`Could not fetch ${modelPath} (HTTP ${r.status})`);
        return r.arrayBuffer();
      }),
      fetch(scalerPath, { cache: 'no-store' }).then((r) => r.json()),
    ]);

    const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new ModelRunner(session, scaler);
  }

  /**
   * @param seqFlat   Float32Array of nWindows * seqLen * nChannels, row-major
   * @returns Float64Array of per-window cheating probabilities
   */
  async score(seqFlat, nWindows, seqLen, nChannels) {
    const { mean, std } = this.scaler;
    if (mean.length !== nChannels || std.length !== nChannels) {
      throw new Error(`Scaler has ${mean.length} channels but the model expects ${nChannels}.`);
    }

    const x = new Float32Array(seqFlat.length);
    for (let i = 0; i < seqFlat.length; i++) {
      const c = i % nChannels;
      x[i] = (seqFlat[i] - mean[c]) / std[c];
    }

    const inputName = this.session.inputNames[0];
    const outputName = this.session.outputNames[0];
    const tensor = new ort.Tensor('float32', x, [nWindows, seqLen, nChannels]);
    const out = await this.session.run({ [inputName]: tensor });

    const logits = out[outputName].data;
    const probs = new Float64Array(nWindows);
    for (let i = 0; i < nWindows; i++) probs[i] = sigmoid(logits[i]);
    return probs;
  }
}
