/**
 * mediapipe-backend.js — browser port of MediaPipeBackend (src/common.py:562-633).
 *
 * Uses the same face_landmarker.task bundle the Python pipeline uses, via the
 * official JS/WASM build, and reproduces the per-frame math exactly:
 *   _eye_gaze / _eye_ear / _mouth_mar  (common.py:453-488)
 *   _euler_from_matrix                 (common.py:545-559)
 *   process()                          (common.py:608-626)
 */

// Pinned to the SAME MediaPipe version the reference pipeline runs
// (requirements.txt installs mediapipe 0.10.35). These are two builds of the
// same graph, and they do drift between releases — this file was on 0.10.14
// against a 0.10.35 Python, 21 releases apart, which showed up as a systematic
// landmark offset and a systematic bias in the final probability. Bump both
// sides together or not at all.
import { FaceLandmarker, FilesetResolver } from
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs';

const WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

// Landmark indices — common.py:166-174. Right eye = person's right.
const RIGHT_EYE = { outer: 33, inner: 133, top: 159, bottom: 145, top2: 158, bottom2: 153, iris: 468 };
const LEFT_EYE = { outer: 263, inner: 362, top: 386, bottom: 374, top2: 385, bottom2: 380, iris: 473 };
const MOUTH = { top: 13, bottom: 14, left: 61, right: 291 };

const px = (lms, i, w, h) => [lms[i].x * w, lms[i].y * h];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** _eye_gaze — common.py:453. Iris centre normalised within the eye aperture. */
function eyeGaze(lms, eye, w, h) {
  const outer = px(lms, eye.outer, w, h);
  const inner = px(lms, eye.inner, w, h);
  const top = px(lms, eye.top, w, h);
  const bottom = px(lms, eye.bottom, w, h);
  const iris = px(lms, eye.iris, w, h);

  const cx = (outer[0] + inner[0]) / 2;
  const cy = (top[1] + bottom[1]) / 2;
  const halfW = Math.abs(outer[0] - inner[0]) / 2;
  const halfH = Math.abs(top[1] - bottom[1]) / 2;
  return [(iris[0] - cx) / (halfW + 1e-6), (iris[1] - cy) / (halfH + 1e-6)];
}

/** _eye_ear — common.py:470. Eye aspect ratio. */
function eyeEar(lms, eye, w, h) {
  const v1 = dist(px(lms, eye.top, w, h), px(lms, eye.bottom, w, h));
  const v2 = dist(px(lms, eye.top2, w, h), px(lms, eye.bottom2, w, h));
  const hd = dist(px(lms, eye.outer, w, h), px(lms, eye.inner, w, h));
  return (v1 + v2) / (2 * hd + 1e-6);
}

/** _mouth_mar — common.py:480. Inner-lip aspect ratio (visual speech signal). */
function mouthMar(lms, w, h) {
  const width = dist(px(lms, MOUTH.right, w, h), px(lms, MOUTH.left, w, h));
  return dist(px(lms, MOUTH.top, w, h), px(lms, MOUTH.bottom, w, h)) / (width + 1e-6);
}

/**
 * Read the 3x3 rotation submatrix out of MediaPipe's 4x4 facial transformation
 * matrix.
 *
 * The Python API hands back a (4,4) array that `_euler_from_matrix` indexes as
 * row-major. The JS API hands back a flat 16-element buffer whose layout is not
 * guaranteed to match, and transposing the rotation silently flips the sign of
 * yaw and roll — which would corrupt two model input channels without any
 * visible error. So detect the layout instead of assuming: the translation
 * component is metric (tens of units) while every rotation entry is within
 * [-1, 1], which makes the two layouts trivially distinguishable.
 */
function rotationFromFlat16(d) {
  const rowMajorT = Math.abs(d[3]) + Math.abs(d[7]) + Math.abs(d[11]);
  const colMajorT = Math.abs(d[12]) + Math.abs(d[13]) + Math.abs(d[14]);
  const at = colMajorT > rowMajorT
    ? (r, c) => d[c * 4 + r]   // column-major
    : (r, c) => d[r * 4 + c];  // row-major
  return at;
}

/** _euler_from_matrix — common.py:545. yaw/pitch/roll in degrees. */
function eulerFromMatrix(flat16) {
  const R = rotationFromFlat16(flat16);
  const deg = (r) => (r * 180) / Math.PI;
  const sy = Math.sqrt(R(0, 0) ** 2 + R(1, 0) ** 2);
  if (sy > 1e-6) {
    return {
      yaw: deg(Math.atan2(R(1, 0), R(0, 0))),
      pitch: deg(Math.atan2(-R(2, 0), sy)),
      roll: deg(Math.atan2(R(2, 1), R(2, 2))),
    };
  }
  return {
    yaw: 0,
    pitch: deg(Math.atan2(-R(2, 0), sy)),
    roll: deg(Math.atan2(-R(1, 2), R(1, 1))),
  };
}

const NO_FACE = {
  face_found: 0, gaze_x: NaN, gaze_y: NaN, yaw: NaN,
  pitch: NaN, roll: NaN, ear: NaN, mouth: NaN, blink: 0,
};

export class MediaPipeBackend {
  constructor(landmarker, cfg, delegate) {
    this.landmarker = landmarker;
    this.cfg = cfg;
    this.delegate = delegate;
  }

  /**
   * Create the landmarker, optionally pinning the inference delegate.
   *
   * `prefer` accepts 'CPU', 'GPU', or null for "GPU, falling back to CPU".
   *
   * The default is CPU, deliberately. The Python reference runs this same .task
   * bundle through TFLite's XNNPACK CPU delegate (common.py:580); the GPU path
   * is a different numerical implementation of the same graph, and measured
   * against Python on real footage it is about twice as far off on every
   * channel:
   *
   *     channel   mean|Δ| GPU   mean|Δ| CPU   (python channel std)
   *     gaze_x       0.01419       0.00778     0.047
   *     gaze_y       0.03064       0.01405     0.064
   *     yaw          0.22212       0.11791     2.84
   *     pitch        0.24915       0.14033     3.35
   *
   * This pipeline is scored against a precision-tuned threshold (0.959) that
   * was fitted on Python's numbers, so agreeing with Python matters more here
   * than landmarking quickly.
   */
  static async create(cfg, modelPath = 'assets/face_landmarker.task', prefer = 'CPU') {
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

    // Fetch the bundle ourselves with caching disabled and hand over the raw
    // bytes (modelAssetBuffer) instead of a path — letting the library fetch
    // by path risks a stale HTTP cache entry surviving a normal reload, same
    // failure mode as the ONNX model.
    const modelBytes = await fetch(modelPath, { cache: 'no-store' }).then((r) => {
      if (!r.ok) throw new Error(`Could not fetch ${modelPath} (HTTP ${r.status})`);
      return r.arrayBuffer();
    });

    const order = prefer === 'CPU' ? ['CPU'] : prefer === 'GPU' ? ['GPU'] : ['GPU', 'CPU'];
    for (const delegate of order) {
      try {
        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetBuffer: new Uint8Array(modelBytes), delegate },
          runningMode: 'IMAGE',
          numFaces: 1,
          outputFacialTransformationMatrixes: true,
          outputFaceBlendshapes: false,
        });
        return new MediaPipeBackend(landmarker, cfg, delegate);
      } catch (err) {
        if (delegate === order[order.length - 1]) throw err;
        console.warn('GPU delegate unavailable, falling back to CPU:', err);
      }
    }
  }

  /** process() — common.py:608. One canvas frame -> the nine per-frame channels. */
  process(canvas) {
    const res = this.landmarker.detect(canvas);
    const lms = res?.faceLandmarks?.[0];
    if (!lms) return { ...NO_FACE };

    const w = canvas.width, h = canvas.height;
    const [gxr, gyr] = eyeGaze(lms, RIGHT_EYE, w, h);
    const [gxl, gyl] = eyeGaze(lms, LEFT_EYE, w, h);
    const clamp2 = (v) => Math.min(Math.max(v, -2), 2);

    const ear = (eyeEar(lms, RIGHT_EYE, w, h) + eyeEar(lms, LEFT_EYE, w, h)) / 2;

    let yaw = 0, pitch = 0, roll = 0;
    const xf = res.facialTransformationMatrixes?.[0];
    if (xf?.data?.length === 16) ({ yaw, pitch, roll } = eulerFromMatrix(xf.data));

    return {
      face_found: 1,
      gaze_x: clamp2((gxr + gxl) / 2),
      gaze_y: clamp2((gyr + gyl) / 2),
      yaw, pitch, roll,
      ear,
      mouth: mouthMar(lms, w, h),
      blink: ear < this.cfg.ear_blink_threshold ? 1 : 0,
    };
  }

  close() {
    try { this.landmarker.close(); } catch { /* already gone */ }
  }
}
