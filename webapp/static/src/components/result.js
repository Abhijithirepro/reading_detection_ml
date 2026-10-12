/* result.js — renders a finished job: verdict, scale, the video review, readouts,
   chart and the window table. User-derived text goes in via textContent only.
   Chart bars and table rows play their window in the review player. */
(function (app) {
  'use strict';

  var LOW_FACE_RATIO = 0.5;
  var el = {};

  /**
   * Creates an element with a class and text.
   * @param {string} tag
   * @param {string} cls
   * @param {string} text
   * @returns {HTMLElement}
   */
  function make(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) {
      node.className = cls;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  /**
   * Verdict word, score and the 0–1 scale.
   * @param {Object} r  payload.result
   * @param {string} source
   */
  function renderVerdict(r, source) {
    var flagged = r.verdict === 'CHEATING';
    el.verdictWord.textContent = flagged ? 'Flagged' : 'Clear';
    el.verdict.classList.toggle('is-flagged', flagged);
    el.verdict.classList.toggle('is-clear', !flagged);
    el.verdictProb.textContent = r.probability.toFixed(3);
    el.verdictThr.textContent = r.threshold.toFixed(3);
    el.verdictSource.textContent = source;
    el.scaleMark.style.left = (Math.max(0, Math.min(1, r.probability)) * 100) + '%';
    el.scaleThr.style.left = (r.threshold * 100) + '%';
    el.panel.classList.toggle('is-flagged-scale', flagged);
  }

  /**
   * The readout grid under the verdict.
   * @param {Object} payload
   */
  function renderReadouts(payload) {
    var f = app.format;
    var r = payload.result;
    var s = payload.stats;
    el.windows.textContent = r.n_windows;
    el.flagged.textContent = r.n_flagged_windows;
    el.span.textContent = f.clock(s.analysed_seconds) + (s.max_seconds ? ' (capped)' : '');
    el.face.textContent = f.percent(s.face_ratio);
    el.time.textContent = f.clock(s.processing_seconds);
    el.speed.textContent = f.fixed(s.speed_x_realtime, 1) + '×';
    el.frames.textContent = s.n_frames.toLocaleString() + ' @ ' + f.fixed(s.sampled_fps, 1) + ' fps';
    el.hw.textContent = payload.runtime.mediapipe + '×' + (payload.runtime.mediapipe_threads || 1) +
      ' + ' + payload.runtime.cnn;
    el.hw.title = 'face tracking on ' + (payload.runtime.mediapipe_threads || 1) + ' CPU threads, CNN on ' +
      payload.runtime.cnn;

    el.faceWarn.hidden = !(s.face_ratio < LOW_FACE_RATIO);
    if (!el.faceWarn.hidden) {
      el.faceWarn.textContent = 'Only ' + f.percent(s.face_ratio) +
        ' of frames have a detected face — check lighting and framing; this result is unreliable.';
    }
  }

  /**
   * Window-by-window table (the accessible fallback for the chart). Rows are
   * focusable and play their window in the review player.
   * @param {Array<Object>} windows
   * @param {number} flagThreshold
   * @param {number} t0  video time of the first analysed frame
   */
  function renderTable(windows, flagThreshold, t0) {
    var mmss = app.format.mmss;
    var tbody = el.table.querySelector('tbody');
    var frag = document.createDocumentFragment();
    var i, w, tr;
    tbody.textContent = '';
    for (i = 0; i < windows.length; i++) {
      w = windows[i];
      tr = document.createElement('tr');
      tr.className = 'is-seekable' + (w.prob >= flagThreshold ? ' is-flagged' : '');
      tr.tabIndex = 0;
      tr.setAttribute('data-i', String(i));
      tr.appendChild(make('td', '', String(i + 1)));
      tr.appendChild(make('td', '', mmss(t0 + w.start_s) + '–' + mmss(t0 + w.end_s)));
      tr.appendChild(make('td', '', w.prob.toFixed(3)));
      tr.appendChild(make('td', '', w.reasons.length ? w.reasons.join(', ') : '—'));
      frag.appendChild(tr);
    }
    tbody.appendChild(frag);
  }

  /**
   * Plays the window a table row stands for.
   * @param {Event} e
   */
  function onTableActivate(e) {
    var row = e.target.closest ? e.target.closest('tr[data-i]') : null;
    if (!row || (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ')) {
      return;
    }
    e.preventDefault();
    app.review.seekWindow(parseInt(row.getAttribute('data-i'), 10));
  }

  /**
   * Renders the whole result block.
   * @param {Object} payload  the server's result payload
   * @param {File|null} file  the uploaded recording, if still in memory (for playback)
   */
  function render(payload, file) {
    var r = payload.result;
    renderVerdict(r, payload.source);
    el.block.hidden = false;           // visible first, so the player can size itself
    app.review.load(payload, file || null);
    renderReadouts(payload);
    app.chart.draw(el.chart, el.chartTip, payload.windows, r.flag_threshold, app.review.seekWindow);
    renderTable(payload.windows, r.flag_threshold, payload.stats.t0 || 0);
  }

  /** Hides the result block and releases the video for a new run. */
  function hide() {
    el.block.hidden = true;
    app.review.reset();
  }

  /** Caches element references. */
  function init() {
    var ids = {
      block: 'result-block', panel: 'panel-live', verdict: 'verdict', verdictWord: 'verdict-word',
      verdictProb: 'verdict-prob', verdictThr: 'verdict-thr', verdictSource: 'verdict-source',
      scaleMark: 'scale-mark', scaleThr: 'scale-thr', faceWarn: 'face-warn',
      windows: 'rs-windows', flagged: 'rs-flagged', span: 'rs-span', face: 'rs-face',
      time: 'rs-time', speed: 'rs-speed', frames: 'rs-frames', hw: 'rs-hw',
      chart: 'chart', chartTip: 'chart-tip', table: 'wtable'
    };
    var key;
    for (key in ids) {
      if (Object.prototype.hasOwnProperty.call(ids, key)) {
        el[key] = document.getElementById(ids[key]);
      }
    }
    if (el.table) {
      el.table.addEventListener('click', onTableActivate);
      el.table.addEventListener('keydown', onTableActivate);
    }
  }

  app.result = { init: init, render: render, hide: hide };
})(window.GazeApp = window.GazeApp || {});
