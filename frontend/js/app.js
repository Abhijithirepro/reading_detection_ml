/**
 * app.js — orchestration for the browser pipeline.
 *
 * Both input modes (upload / webcam) drive the same session.js Session and
 * render through the same renderSnapshot()/renderResult() pair — a verdict
 * appears as soon as the first 20s window scores and keeps updating roughly
 * every 10s after that, rather than only once a run is fully complete.
 */

import { MediaPipeBackend } from './mediapipe-backend.js';
import { ModelRunner } from './model-runner.js';
import { openCamera, stopCamera } from './live-source.js';
import { runFileSession, runLiveSession } from './session.js';
import { windowReasons, mmss } from './aggregate.js';

const $ = (id) => document.getElementById(id);

const el = {
  modeUploadBtn: $('mode-upload-btn'), modeWebcamBtn: $('mode-webcam-btn'),
  intakeUpload: $('intake-upload'), intakeWebcam: $('intake-webcam'),
  drop: $('drop'), file: $('file'), run: $('run'), maxmin: $('maxmin'), filename: $('filename'),
  camLive: $('cam-live'), camPreview: $('cam-preview'), camError: $('cam-error'),
  camStart: $('cam-start'),
  panelIntake: $('panel-intake'), panelLive: $('panel-live'), panelError: $('panel-error'),
  liveBadge: $('live-badge'), liveText: $('live-text'),
  liveElapsed: $('live-elapsed'), liveFrames: $('live-frames'), liveFace: $('live-face'),
  liveEtaCell: $('live-eta-cell'), liveEta: $('live-eta'),
  meter: $('meter'), meterFill: $('meter-fill'),
  liveWarn: $('live-warn'), liveError: $('live-error'),
  liveStop: $('live-stop'), liveAgain: $('live-again'), download: $('download'),
  warming: $('warming'), resultBlock: $('result-block'),
  retry: $('retry'), errorMsg: $('error-msg'),
  verdict: $('verdict'), verdictWord: $('verdict-word'), verdictProb: $('verdict-prob'), verdictThr: $('verdict-thr'),
  scaleMark: $('scale-mark'), scaleThr: $('scale-thr'),
  rsWindows: $('rs-windows'), rsFlagged: $('rs-flagged'), rsSpan: $('rs-span'), rsFace: $('rs-face'),
  rsTime: $('rs-time'), rsRate: $('rs-rate'),
  chart: $('chart'), chartTip: $('chart-tip'), segments: $('segments'), segmentsEmpty: $('segments-empty'),
  wtable: $('wtable'),
  specKind: $('spec-kind'), specThreshold: $('spec-threshold'), specWindow: $('spec-window'),
  specChannels: $('spec-channels'), specRuntime: $('spec-runtime'),
};

const state = {
  cfg: null, artifact: null, backend: null, runner: null,
  mode: 'upload', file: null, camStream: null, camVideo: null,
  controller: null, lastResult: null,
};

const LOW_FACE_RATIO = 0.5;
const MIN_FRAMES_FOR_WARNING = 30;

// ── boot ────────────────────────────────────────────────────────────────

async function boot() {
  try {
    const [cfg, artifact] = await Promise.all([
      fetch('assets/config.json', { cache: 'no-store' }).then((r) => r.json()),
      fetch('assets/model.json', { cache: 'no-store' }).then((r) => r.json()),
    ]);
    state.cfg = cfg;
    state.artifact = artifact;

    el.specKind.textContent = artifact.kind;
    el.specThreshold.textContent = artifact.threshold.toFixed(3);
    el.specWindow.textContent = `${cfg.window_seconds}s / ${cfg.stride_seconds}s`;
    el.specChannels.textContent = `${artifact.seq_channels.length}ch @ ${cfg.target_fps}fps`;

    const [backend, runner] = await Promise.all([
      MediaPipeBackend.create(cfg),
      ModelRunner.create(),
    ]);
    state.backend = backend;
    state.runner = runner;

    el.specRuntime.textContent = `${backend.delegate.toLowerCase()} + wasm`;
    updateReadiness();
  } catch (err) {
    console.error(err);
    el.specRuntime.textContent = 'failed';
    fail(`Could not initialise the model: ${err.message}. Serve this folder over HTTP — opening index.html as a file:// URL blocks the model and WASM fetches.`);
  }
}

function modelReady() { return !!(state.backend && state.runner); }

function updateReadiness() {
  el.run.disabled = !(modelReady() && state.file);
  el.camStart.disabled = !modelReady();
}

// ── mode toggle ─────────────────────────────────────────────────────────

function setMode(mode) {
  if (state.mode === mode) return;
  if (state.mode === 'webcam' && state.camStream && !state.controller) {
    // Defensive: a camera stream open with no active session shouldn't
    // normally happen (Start opens the camera and begins the session in one
    // click), but release it if it's ever the case.
    stopCamera(state.camStream);
    resetCameraUI();
  }
  state.mode = mode;
  el.modeUploadBtn.classList.toggle('is-active', mode === 'upload');
  el.modeWebcamBtn.classList.toggle('is-active', mode === 'webcam');
  el.modeUploadBtn.setAttribute('aria-selected', String(mode === 'upload'));
  el.modeWebcamBtn.setAttribute('aria-selected', String(mode === 'webcam'));
  el.intakeUpload.hidden = mode !== 'upload';
  el.intakeWebcam.hidden = mode !== 'webcam';
}

el.modeUploadBtn.addEventListener('click', () => setMode('upload'));
el.modeWebcamBtn.addEventListener('click', () => setMode('webcam'));

// ── upload intake ───────────────────────────────────────────────────────

function acceptFile(file) {
  if (!file) return;
  state.file = file;
  el.filename.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  el.drop.classList.add('is-loaded');
  updateReadiness();
}

el.drop.addEventListener('click', () => el.file.click());
el.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); }
});
el.file.addEventListener('change', () => acceptFile(el.file.files[0]));

['dragenter', 'dragover'].forEach((t) => el.drop.addEventListener(t, (e) => {
  e.preventDefault(); el.drop.classList.add('is-over');
}));
['dragleave', 'drop'].forEach((t) => el.drop.addEventListener(t, (e) => {
  e.preventDefault(); el.drop.classList.remove('is-over');
}));
el.drop.addEventListener('drop', (e) => acceptFile(e.dataTransfer?.files?.[0]));

el.run.addEventListener('click', () => startAnalysis('upload'));

// ── webcam intake ───────────────────────────────────────────────────────

const CAM_ERROR_COPY = {
  NotAllowedError: 'Camera permission was denied — check your browser\'s site settings and try again.',
  PermissionDeniedError: 'Camera permission was denied — check your browser\'s site settings and try again.',
  NotFoundError: 'No camera device was found on this machine.',
  OverconstrainedError: 'No camera matched the requested resolution.',
  NotReadableError: 'The camera is already in use by another application.',
};

function resetCameraUI() {
  state.camStream = null;
  state.camVideo = null;
  el.camLive.hidden = true;
  el.camPreview.srcObject = null;
}

/**
 * One click does both: request the camera and, the instant it's live, start
 * the analysis session — no separate "enable" step. The live self-view moves
 * into the live-analysis panel and stays visible for the whole session;
 * ending it (the live-stop button, labelled "End" in this mode) is what
 * finalizes and shows the result.
 */
el.camStart.addEventListener('click', async () => {
  el.camError.hidden = true;
  el.camStart.disabled = true;
  try {
    const { stream, video } = await openCamera();
    state.camStream = stream;
    state.camVideo = video;             // hidden capture source (session.js draws from this)
    el.camPreview.srcObject = stream;    // visible self-view, same stream
    await startAnalysis('webcam');
  } catch (err) {
    const msg = CAM_ERROR_COPY[err.name] || err.message || String(err);
    el.camError.textContent = msg;
    el.camError.hidden = false;
  } finally {
    el.camStart.disabled = false;
  }
});

// ── panel switching ─────────────────────────────────────────────────────

function show(panel) {
  for (const p of [el.panelIntake, el.panelLive, el.panelError]) p.hidden = p !== panel;
}

function fail(message) {
  el.errorMsg.textContent = message;
  show(el.panelError);
}

function resetToIntake() {
  state.file = null;
  el.file.value = '';
  el.filename.textContent = '';
  el.drop.classList.remove('is-loaded');
  if (state.camStream) { stopCamera(state.camStream); resetCameraUI(); }
  state.controller = null;
  state.lastResult = null;
  updateReadiness();
  show(el.panelIntake);
}

el.retry.addEventListener('click', resetToIntake);
el.liveAgain.addEventListener('click', resetToIntake);
el.liveStop.addEventListener('click', () => state.controller?.abort());

window.addEventListener('pagehide', () => { if (state.camStream) stopCamera(state.camStream); });

// ── run ─────────────────────────────────────────────────────────────────

const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

async function startAnalysis(mode) {
  const { cfg, artifact, backend, runner } = state;
  const controller = new AbortController();
  state.controller = controller;
  state.lastResult = null;

  el.liveEtaCell.hidden = mode !== 'upload';
  el.meter.hidden = mode !== 'upload';
  el.camLive.hidden = mode !== 'webcam';
  el.liveStop.textContent = mode === 'webcam' ? 'End' : 'Stop';
  el.liveStop.hidden = false;
  el.liveAgain.hidden = true;
  el.download.hidden = true;
  el.liveWarn.hidden = true;
  el.liveError.hidden = true;
  el.warming.hidden = false;
  el.resultBlock.hidden = true;
  renderStatus({ status: 'warming', nFrames: 0, faceRatio: 0, elapsedSeconds: 0, progress: null });
  show(el.panelLive);

  try {
    if (mode === 'upload') {
      const minutes = parseFloat(el.maxmin.value);
      const maxSeconds = Number.isFinite(minutes) && minutes > 0 ? minutes * 60 : null;
      await runFileSession(state.file, cfg, artifact, backend, runner, {
        maxSeconds, signal: controller.signal, onSnapshot: (snap) => renderSnapshot(snap, mode),
      });
    } else {
      await runLiveSession(state.camVideo, cfg, artifact, backend, runner, {
        signal: controller.signal, onSnapshot: (snap) => renderSnapshot(snap, mode),
      });
      // The session is over one way or another (Stop, or the track ended) —
      // release the camera hardware; a new session re-requests it.
      stopCamera(state.camStream);
      resetCameraUI();
    }
  } catch (err) {
    console.error(err);
    el.liveError.textContent = err.message || String(err);
    el.liveError.hidden = false;
    el.liveStop.hidden = true;
    el.liveAgain.hidden = false;
    if (mode === 'webcam' && state.camStream) { stopCamera(state.camStream); resetCameraUI(); }
  } finally {
    state.controller = null;
  }
}

// ── snapshot rendering ──────────────────────────────────────────────────

function renderStatus(snap) {
  const badges = {
    warming: ['Warming up', 'first reading in ~20s of usable signal'],
    scoring: ['Live', 'updating roughly every 10s'],
    finished: [snap.stoppedEarly ? 'Stopped' : 'Finished',
      snap.stoppedEarly ? 'stopped early — showing everything captured' : 'session complete'],
    'too-short': ['Too short', `needed at least ${state.cfg.window_seconds}s of usable signal`],
    error: ['Error', 'see the message below'],
  };
  const [badge, text] = badges[snap.status] || ['—', ''];
  el.liveBadge.textContent = badge;
  el.liveText.textContent = text;
  el.liveBadge.className = `live-status__badge live-status__badge--${snap.status}`;

  el.liveElapsed.textContent = clock(snap.elapsedSeconds);
  el.liveFrames.textContent = snap.nFrames.toLocaleString();
  el.liveFace.textContent = snap.nFrames ? `${Math.round(snap.faceRatio * 100)}%` : '—';

  if (snap.progress) {
    // Progress is tracked in media time, not frame count: the sampler follows
    // the video's own frame timing (frame-sampler.js), so how many frames a
    // recording will yield isn't known until it has been walked.
    const { index, seconds, endAt } = snap.progress;
    // endAt is null unless the user capped the run: no container here reports a
    // duration worth trusting (frame-sampler.js), so an uncapped run shows how
    // much footage has been scored rather than a fake percentage.
    if (endAt) {
      const done = Math.min(1, seconds / endAt);
      const pct = done * 100;
      el.meterFill.style.width = `${pct}%`;
      el.meter.setAttribute('aria-valuenow', String(Math.round(pct)));
      const eta = done > 0 ? (snap.elapsedSeconds / done) * (1 - done) : 0;
      el.liveEta.textContent = index > 30 && done > 0 ? clock(eta) : '—';
    } else {
      el.meterFill.style.width = '100%';
      el.meter.removeAttribute('aria-valuenow');
      el.liveEta.textContent = `${clock(seconds)} scored`;
    }
  }

  const warnWorthy = snap.nFrames >= MIN_FRAMES_FOR_WARNING && snap.faceRatio < LOW_FACE_RATIO;
  el.liveWarn.hidden = !warnWorthy;
  if (warnWorthy) {
    el.liveWarn.textContent =
      `Only ${Math.round(snap.faceRatio * 100)}% of frames have a detected face — ` +
      `check lighting and framing; results below are unreliable while this stays low.`;
  }

  const terminal = snap.status === 'finished' || snap.status === 'too-short' || snap.status === 'error';
  el.liveStop.hidden = terminal;
  el.liveAgain.hidden = !terminal;
  el.download.hidden = !(terminal && snap.result);
}

function renderSnapshot(snap, mode) {
  renderStatus(snap);

  if (!snap.result) {
    el.warming.hidden = false;
    el.resultBlock.hidden = true;
    return;
  }
  el.warming.hidden = true;
  el.resultBlock.hidden = false;

  const result = { ...snap.result };
  const lastWindow = snap.windows[snap.windows.length - 1];
  result.source = mode === 'upload' ? state.file.name : 'webcam-session';
  result.analysed_seconds = lastWindow ? lastWindow.win_end_s : 0;
  result.face_ratio = +snap.faceRatio.toFixed(3);
  result.processing_seconds = +snap.elapsedSeconds.toFixed(1);
  result.seconds_per_analysed_minute = result.analysed_seconds > 0
    ? +(result.processing_seconds / (result.analysed_seconds / 60)).toFixed(1)
    : null;

  state.lastResult = result;
  renderResult(result, snap.windows, snap.probs);
}

// ── render ──────────────────────────────────────────────────────────────

function renderResult(result, windows, probs) {
  const flagged = result.verdict === 'CHEATING';

  el.verdictWord.textContent = flagged ? 'Flagged' : 'Clear';
  el.verdict.classList.toggle('is-flagged', flagged);
  el.verdict.classList.toggle('is-clear', !flagged);
  el.verdictProb.textContent = result.probability.toFixed(3);
  el.verdictThr.textContent = result.threshold.toFixed(3);

  el.scaleMark.style.left = `${result.probability * 100}%`;
  el.scaleThr.style.left = `${result.threshold * 100}%`;
  el.panelLive.classList.toggle('is-flagged-scale', flagged);

  el.rsWindows.textContent = result.n_windows;
  el.rsFlagged.textContent = result.n_flagged_windows;
  el.rsSpan.textContent = clock(result.analysed_seconds);
  el.rsFace.textContent = `${Math.round(result.face_ratio * 100)}%`;
  el.rsTime.textContent = clock(result.processing_seconds);
  el.rsRate.textContent = result.seconds_per_analysed_minute != null
    ? `${result.seconds_per_analysed_minute.toFixed(1)}s`
    : '—';

  drawChart(windows, probs, result.threshold);

  el.segments.innerHTML = '';
  el.segmentsEmpty.hidden = result.flagged_segments.length > 0;
  for (const s of result.flagged_segments) {
    const li = document.createElement('li');
    li.className = 'seg';
    li.innerHTML =
      `<span class="seg__range">${s.range}</span>` +
      `<span class="seg__reason">${s.reason}</span>` +
      `<span class="seg__peak">peak ${s.peak_prob.toFixed(3)}</span>`;
    el.segments.appendChild(li);
  }

  const tbody = el.wtable.querySelector('tbody');
  tbody.innerHTML = '';
  windows.forEach((w, i) => {
    const tr = document.createElement('tr');
    if (probs[i] >= result.flag_threshold) tr.className = 'is-flagged';
    const reasons = windowReasons(w).join(', ') || '—';
    tr.innerHTML =
      `<td>${i + 1}</td>` +
      `<td>${mmss(w.win_start_s)}–${mmss(w.win_end_s)}</td>` +
      `<td>${probs[i].toFixed(3)}</td>` +
      `<td>${reasons}</td>`;
    tbody.appendChild(tr);
  });
}

/**
 * Per-window probability over time.
 *
 * One measure on one axis: probability. Colour carries a state (below vs above
 * the flag threshold) using the reserved status red, not a series hue, and the
 * legend plus the table view keep that distinction off colour alone.
 */
function drawChart(windows, probs, threshold) {
  const W = 1000, H = 210;
  const PAD = { t: 10, r: 8, b: 26, l: 38 };
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;

  const n = windows.length;
  const slot = plotW / n;
  const barW = Math.max(1, slot - (slot > 5 ? 2 : 0.5));   // 2px surface gap when there is room
  const y = (p) => PAD.t + plotH * (1 - p);

  const parts = [];
  parts.push(`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Suspicion score for each ${windows.length}-window span of the recording.">`);

  for (const gv of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(
      `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${y(gv)}" y2="${y(gv)}" stroke="var(--grid)" stroke-width="1"/>` +
      `<text x="${PAD.l - 8}" y="${y(gv) + 3.5}" text-anchor="end" font-size="10" fill="var(--ink-mute)" font-family="var(--mono)">${gv.toFixed(2)}</text>`
    );
  }

  windows.forEach((w, i) => {
    const p = probs[i];
    const hit = p >= threshold;
    const h = Math.max(1.5, plotH * p);
    const x = PAD.l + i * slot + (slot - barW) / 2;
    parts.push(
      `<rect class="bar" x="${x.toFixed(2)}" y="${(PAD.t + plotH - h).toFixed(2)}" ` +
      `width="${barW.toFixed(2)}" height="${h.toFixed(2)}" rx="${barW > 5 ? 2 : 0}" ` +
      `fill="${hit ? 'var(--series-flag)' : 'var(--series-base)'}" ` +
      `data-i="${i}" data-p="${p.toFixed(3)}" ` +
      `data-r="${mmss(w.win_start_s)}–${mmss(w.win_end_s)}"><title>${mmss(w.win_start_s)}–${mmss(w.win_end_s)}: ${p.toFixed(3)}</title></rect>`
    );
  });

  parts.push(
    `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${y(threshold)}" y2="${y(threshold)}" ` +
    `stroke="var(--amber)" stroke-width="2" stroke-dasharray="6 4"/>`
  );
  parts.push(
    `<line x1="${PAD.l}" x2="${W - PAD.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--axis)" stroke-width="1"/>`
  );

  const last = windows[n - 1];
  parts.push(
    `<text x="${PAD.l}" y="${H - 8}" font-size="10" fill="var(--ink-mute)" font-family="var(--mono)">00:00</text>` +
    `<text x="${W - PAD.r}" y="${H - 8}" text-anchor="end" font-size="10" fill="var(--ink-mute)" font-family="var(--mono)">${mmss(last.win_end_s)}</text>`
  );
  parts.push('</svg>');
  el.chart.innerHTML = parts.join('');

  const svg = el.chart.querySelector('svg');
  svg.addEventListener('mousemove', (e) => {
    const bar = e.target.closest('rect.bar');
    if (!bar) { el.chartTip.hidden = true; return; }
    const box = el.chart.getBoundingClientRect();
    el.chartTip.hidden = false;
    el.chartTip.innerHTML = `<b>${bar.dataset.p}</b><span>${bar.dataset.r}</span>`;
    el.chartTip.style.left = `${e.clientX - box.left}px`;
    el.chartTip.style.top = `${e.clientY - box.top}px`;
  });
  svg.addEventListener('mouseleave', () => { el.chartTip.hidden = true; });
}

el.download.addEventListener('click', () => {
  if (!state.lastResult) return;
  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${state.lastResult.source.replace(/\.[^.]+$/, '')}__prediction.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

boot();
