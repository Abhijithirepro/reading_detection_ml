/**
 * live-source.js — webcam capture.
 *
 * Mirrors frame-sampler.js's onFrame(canvas, t) callback shape exactly, so every
 * downstream consumer (mediapipe-backend.js, pipeline.js, session.js) is 100%
 * shared code between an uploaded file and a live camera — only the frame
 * *source* differs.
 *
 * Like frame-sampler.js, this applies extract_features.py:114's keep rule to the
 * frames' OWN presentation timestamps rather than to a wall-clock grid:
 *
 *     if (t + 1e-9 >= nextT) { keep(t); nextT = t + 1 / target_fps; }
 *
 * That matters for the same reason it matters offline. Re-basing `nextT` on the
 * kept frame's timestamp makes the schedule drift to the camera's own frame
 * grid — 12.5fps on a 25fps camera, 15fps on a 30fps one — which is the sampling
 * density the model was trained against. A fixed 15fps wall-clock schedule, what
 * this file used to do, feeds it a signal no training video ever had.
 *
 * Unlike the offline path there is no pausing: a live stream can't be held while
 * MediaPipe runs, so if the tab falls behind, frames are missed rather than
 * queued. That is unavoidable for live capture and is what interpolate_gaps /
 * resample_uniform already exist to absorb.
 */

/** Request the camera and resolve once its preview has usable dimensions. */
export async function openCamera(constraints) {
  constraints = constraints || {
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: false,
  };
  // NotAllowedError / NotFoundError / OverconstrainedError / NotReadableError
  // propagate untouched — the caller maps them to user-facing copy.
  const stream = await navigator.mediaDevices.getUserMedia(constraints);

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;

  await new Promise((resolve, reject) => {
    video.addEventListener('loadedmetadata', () => resolve(), { once: true });
    video.addEventListener('error', () => reject(new Error('Could not start the camera preview.')), { once: true });
  });
  await video.play();

  return { stream, video };
}

/** Release every track so the OS camera indicator turns off. */
export function stopCamera(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
}

/**
 * Start a real-time capture loop over a live <video> already playing a camera
 * stream. Returns { stop() } — open-ended, not a Promise, since a live session
 * has no known end until the caller stops it.
 *
 * @param opts { onProgress, onTrackEnded }
 */
export function startLiveCapture(video, cfg, onFrame, opts) {
  opts = opts || {};
  const onProgress = opts.onProgress || null;
  const onTrackEnded = opts.onTrackEnded || null;

  if (typeof video.requestVideoFrameCallback !== 'function') {
    throw new Error(
      'This browser cannot report per-frame timestamps (requestVideoFrameCallback). '
      + 'Use Chrome — a live session has to follow the camera\'s own frame timing '
      + 'to score the same way the reference pipeline does.'
    );
  }

  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) throw new Error('Camera reports no frame dimensions yet.');

  // Native resolution, matching frame-sampler.js — iris offset within the eye
  // aperture is sub-pixel, so downscaling costs real gaze precision.
  const canvas = document.createElement('canvas');
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const targetDt = 1 / cfg.target_fps;
  let nextT = 0.0;
  let started = false;
  let stopped = false;
  let frameCount = 0;

  const track = video.srcObject?.getVideoTracks?.()[0];
  if (track && onTrackEnded) track.addEventListener('ended', onTrackEnded);

  function tick(_now, meta) {
    if (stopped) return;
    const t = meta.mediaTime;

    // A live timeline starts wherever the element happens to be, so anchor the
    // first boundary on the first frame actually seen rather than on zero.
    if (!started) { started = true; nextT = t; }

    if (t + 1e-9 >= nextT) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      frameCount++;
      onFrame(canvas, t);
      nextT = t + targetDt;
      if (onProgress) onProgress({ frameCount, seconds: t });
    }
    video.requestVideoFrameCallback(tick);
  }

  video.requestVideoFrameCallback(tick);

  return {
    stop() {
      stopped = true;
      if (track && onTrackEnded) track.removeEventListener('ended', onTrackEnded);
    },
  };
}
