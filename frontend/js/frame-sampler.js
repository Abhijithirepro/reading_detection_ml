/**
 * frame-sampler.js — browser port of the frame loop in
 * src/extract_features.py:94-135.
 *
 * PARITY CONTRACT
 * ---------------
 * The Python CLI is the reference implementation: its verdicts are what the
 * model's threshold was tuned against and what has been signed off for
 * production. This file therefore reproduces `extract_one`'s sampling loop
 * *exactly*, quirks included — it is not an idealised reimplementation.
 *
 * Two properties of that loop are load-bearing and were previously not
 * reproduced here, each large enough on its own to flip a verdict:
 *
 * 1. SEQUENTIAL DECODE, NOT A FIXED GRID. Python walks the file frame by frame
 *    with OpenCV and keeps a frame when its *own* presentation timestamp has
 *    crossed the next boundary, then re-bases the boundary on that timestamp:
 *
 *        if t + 1e-9 >= next_t:  keep(t);  next_t = t + 1/target_fps
 *
 *    Because `next_t` is re-based on the kept frame's actual time rather than
 *    advanced along a fixed grid, the schedule *drifts*: on a 25fps source the
 *    66.7ms boundary always lands mid-frame, so the next kept frame is at 80ms
 *    and the period locks to 80ms — an effective 12.5fps, not 15. Every video
 *    in the training set was sampled this way (12.5fps on 25fps sources, ~7.8fps
 *    on the variable-rate webm ones), so the model has never seen a true 15fps
 *    signal. Seeking to i/15 fed it a denser, differently-phased signal than
 *    anything it was trained on and moved window probabilities by up to 0.07.
 *
 *    (The drift is a bug in extract_features.py, but fixing it there would
 *    invalidate every extracted feature, the trained model and the reports
 *    already reviewed. It has to be fixed on both sides at once, with a
 *    re-extract and retrain. Until then, the browser matches Python.)
 *
 * 2. NATIVE RESOLUTION. Python hands MediaPipe the frame as decoded. This file
 *    used to cap the long edge at 640px on the reasoning that scale cancels out
 *    of the per-frame formulas. It does cancel out of the arithmetic, but not
 *    out of *landmark precision*: iris offset within the eye aperture is a
 *    sub-pixel quantity, and halving a 720p frame shifted gaze_y by half that
 *    channel's own standard deviation.
 *
 * Sequential decode is reproduced with requestVideoFrameCallback, which reports
 * each presented frame's true presentation timestamp (`mediaTime`) — the direct
 * equivalent of OpenCV's CAP_PROP_POS_MSEC. Playback is paused for the duration
 * of each inference so decoding can never outrun scoring; see `assertNoSkips`
 * for what happens if the decoder drops a frame anyway.
 */

/** Matches Python's `if t + 1e-9 >= next_t` (extract_features.py:114). */
const T_EPS = 1e-9;

/**
 * Missed-frame detection uses rVFC's own `presentedFrames` counter, which
 * increments once per frame the compositor is handed. If it advances by more
 * than one between two of our callbacks, a frame went past without us seeing
 * it, and the run is short of footage Python had.
 *
 * Two other signals were tried and are both wrong, for opposite reasons:
 *
 *   - Gaps between mediaTime values. Unsound: a gap in the frames we're shown
 *     is indistinguishable, from timing alone, from a gap in the SOURCE. These
 *     recordings are full of real ones — Shashank_Verma.webm jumps 2.544s
 *     between its 3rd and 4th frame and ffmpeg reports the identical jump — so
 *     this rejected perfectly good videos as corrupt.
 *
 *   - getVideoPlaybackQuality().droppedVideoFrames. It counts frames not
 *     COMPOSITED, which is the normal consequence of pausing on every kept
 *     frame: measured on three videos whose sampling matches Python frame for
 *     frame, it reported 782, 630 and 742 dropped out of ~1800. It describes
 *     rendering smoothness, which we don't care about, not frame loss.
 *
 * Genuine source gaps need no handling here: Python walks straight through them
 * and interpolate_gaps / resample_uniform absorb them on both sides.
 */

export function loadVideo(file) {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.src = URL.createObjectURL(file);
    video.addEventListener('loadedmetadata', () => resolve(video), { once: true });
    video.addEventListener('error', () => reject(new Error('Could not decode this video file.')), { once: true });
  });
}

/**
 * Sample `video` on Python's schedule and hand each kept frame to
 * `onFrame(canvas, t)`, where `t` is the frame's true presentation timestamp.
 * Returns { times, missed } — `times` is the list of kept timestamps, `missed`
 * the number of presented frames that slipped past our callback (see
 * assertNoMissedFrames).
 *
 * @param {object} opts  { maxSeconds, signal, onProgress }
 */
export async function sampleFrames(video, cfg, onFrame, opts = {}) {
  const { maxSeconds = null, signal = null, onProgress = null } = opts;

  if (typeof video.requestVideoFrameCallback !== 'function') {
    throw new Error(
      'This browser cannot report per-frame timestamps '
      + '(requestVideoFrameCallback). Use Chrome — scoring here has to follow '
      + 'the video\'s own frame timing to match the reference pipeline.'
    );
  }

  // `video.duration` is NOT trustworthy on these recordings and must never gate
  // sampling. The same bogus webm headers extract_features.py:85 warns about
  // show up here: one 15-minute file reports 3081s, another reports Infinity.
  // Python never consults duration either — it decodes until the stream ends —
  // so the stop rule is Python's rule: run to `ended`, or past --max-seconds.
  const endAt = Number.isFinite(maxSeconds) && maxSeconds > 0 ? maxSeconds : Infinity;
  // Nothing here reports a trustworthy total, so the progress bar is driven by
  // media time processed and only shows a percentage when the user capped the
  // run ("analyse first N minutes"). null = indeterminate.
  const progressEnd = Number.isFinite(endAt) ? endAt : null;

  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) throw new Error('Video reports no dimensions.');

  // Native resolution — see the parity contract above.
  const canvas = document.createElement('canvas');
  canvas.width = vw;
  canvas.height = vh;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  const targetDt = 1 / cfg.target_fps;
  const times = [];

  let nextT = 0.0;          // Python's `next_t`
  let kept = 0;
  let prevPresentedFrames = null;
  let missed = 0;

  video.muted = true;
  video.playbackRate = 1;
  if (video.currentTime !== 0) video.currentTime = 0;

  await new Promise((resolve, reject) => {
    let settled = false;
    const finish = (err) => {
      if (settled) return;
      settled = true;
      video.pause();
      video.removeEventListener('ended', onEnded);
      video.removeEventListener('error', onError);
      err ? reject(err) : resolve();
    };
    const onEnded = () => finish();
    const onError = () => finish(new Error('Playback failed while sampling frames.'));
    video.addEventListener('ended', onEnded);
    video.addEventListener('error', onError);

    const resume = () => {
      // Re-arm before resuming so no frame can present un-observed.
      video.requestVideoFrameCallback(tick);
      const p = video.play();
      if (p && typeof p.catch === 'function') {
        p.catch((err) => {
          // Our own pause() on the next kept frame interrupts the pending play()
          // request, and the spec says that rejects with AbortError. That is the
          // normal steady state of this loop, not a failure — and emphatically
          // not the user pressing Stop, which is what session.js concludes from
          // an AbortError escaping this function.
          if (err?.name === 'AbortError') return;
          finish(err);
        });
      }
    };

    function tick(_now, meta) {
      if (settled) return;
      if (signal?.aborted) return finish(new DOMException('Aborted', 'AbortError'));

      if (prevPresentedFrames !== null) {
        missed += Math.max(0, meta.presentedFrames - prevPresentedFrames - 1);
      }
      prevPresentedFrames = meta.presentedFrames;

      const t = meta.mediaTime;
      if (t > endAt) return finish();

      if (t + T_EPS >= nextT) {
        // Stop the clock: nothing may present while MediaPipe runs.
        video.pause();
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        onFrame(canvas, t);
        times.push(t);
        nextT = t + targetDt;
        kept++;
        if (onProgress && kept % 15 === 0) {
          onProgress({ index: kept, seconds: t, endAt: progressEnd });
        }
        resume();
        return;
      }

      video.requestVideoFrameCallback(tick);
    }

    resume();
  });

  const last = times[times.length - 1] ?? 0;
  onProgress?.({ index: kept, seconds: last, endAt: progressEnd ?? last, done: true });
  return { times, missed };
}

/**
 * Throw if any presented frame slipped past the sampling callback. Any such
 * frame is one the reference pipeline saw and we did not, so the run cannot be
 * claimed to match it.
 *
 * Called by the caller rather than inline so an aborted run doesn't raise: a
 * partial result the user asked to stop is fine, a silently-wrong complete one
 * is not.
 */
export function assertNoMissedFrames(missed) {
  if (!missed) return;
  throw new Error(
    `${missed} video frame(s) went past without being scored, so this run saw less `
    + 'footage than the reference pipeline and its score would not match. '
    + 'Close other tabs and run it again.'
  );
}
