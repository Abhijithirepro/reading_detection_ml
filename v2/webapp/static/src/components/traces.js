/* traces.js (v2) — the detailed report's live part: the signals the reading model
   looks at, drawn around the playhead and redrawn as the video plays.
     eyes left <-> right   eye_h (blendshapes, calibrated to the person's median)
     eyes up <-> down      eye_v, with a tick per blink
     head turn             yaw
     mouth                 mouth openness, green band = speaking
     P(reading)            per-second reading probability, dashed = threshold
   Detected reading segments are shaded red across every row; reviewer spans sit
   as an amber band on top. Click anywhere to play from there. Data comes from
   payload.traces_url (5 samples a second, written by the server / library.py). */
(function (app) {
  'use strict';

  var ROW_H = 58;
  var ROW_GAP = 6;
  var AXIS_H = 22;
  var LABEL_PAD = 6;
  var ROWS = [
    { key: 'eye_h', label: 'eyes left ↔ right' },
    { key: 'eye_v', label: 'eyes up ↕ down · ticks = blinks', blinks: true },
    { key: 'yaw', label: 'head turn' },
    { key: 'mouth', label: 'mouth · green = speaking', speaking: true },
    { key: 'prob', label: 'P(reading) per second', fixed: [0, 1] }
  ];
  var TICK_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800];

  var el = {};
  var state = {
    payload: null, data: null, ranges: {}, span: 120, t: 0, drawnT: -1, drawnW: 0,
    duration: 0, colors: null, token: 0
  };

  /** Chart colours from the page's tokens (read once; the page is dark-only). */
  function colors() {
    var cs = window.getComputedStyle(document.documentElement);
    function v(name, fallback) {
      return (cs.getPropertyValue(name) || '').trim() || fallback;
    }
    return {
      plane: v('--plane', '#0d0d0d'), hairline: v('--hairline', '#2c2c2a'),
      ink: v('--ink', '#ffffff'), ink2: v('--ink-2', '#c3c2b7'), mute: v('--ink-mute', '#898781'),
      base: v('--series-base', '#3987e5'), flag: v('--series-flag', '#d03b3b'),
      amber: v('--amber', '#f0a726'), good: v('--good', '#0ca30c'), mono: v('--mono', 'monospace')
    };
  }

  /**
   * Robust display range of a channel over the whole recording (2nd-98th pct,
   * padded), so the scale holds still while the window scrolls.
   * @param {Array<number|null>} xs
   * @returns {Array<number>}
   */
  function robustRange(xs) {
    var vals = [];
    var step = Math.max(1, Math.floor(xs.length / 4000));
    var i, lo, hi, pad;
    for (i = 0; i < xs.length; i += step) {
      if (xs[i] !== null && isFinite(xs[i])) {
        vals.push(xs[i]);
      }
    }
    if (!vals.length) {
      return [-1, 1];
    }
    vals.sort(function (a, b) { return a - b; });
    lo = vals[Math.floor(0.02 * (vals.length - 1))];
    hi = vals[Math.ceil(0.98 * (vals.length - 1))];
    if (hi - lo < 1e-6) {
      lo -= 0.5;
      hi += 0.5;
    }
    pad = 0.12 * (hi - lo);
    return [lo - pad, hi + pad];
  }

  /**
   * A tick step giving roughly 6-10 labels across `span` seconds.
   * @param {number} span
   * @returns {number}
   */
  function tickStep(span) {
    var i;
    for (i = 0; i < TICK_STEPS.length; i++) {
      if (span / TICK_STEPS[i] <= 10) {
        return TICK_STEPS[i];
      }
    }
    return TICK_STEPS[TICK_STEPS.length - 1];
  }

  /**
   * Left edge of the visible window: centred on `t`, clamped to the recording.
   * @param {number} t
   * @returns {number}
   */
  function windowStart(t) {
    var D = state.duration;
    var S = state.span;
    return Math.max(0, Math.min(t - S / 2, Math.max(0, D - S)));
  }

  /** Sizes the canvas backing store to its box and the device pixel ratio. */
  function fit() {
    var dpr = window.devicePixelRatio || 1;
    var w = el.plot.clientWidth;
    var h = ROWS.length * ROW_H + (ROWS.length - 1) * ROW_GAP + AXIS_H;
    el.canvas.style.height = h + 'px';
    el.canvas.width = Math.round(w * dpr);
    el.canvas.height = Math.round(h * dpr);
    return { w: w, h: h, dpr: dpr };
  }

  /**
   * Draws the window around time `t`.
   * @param {number} t
   */
  function draw(t) {
    var c = state.colors;
    var d = state.data;
    var r = state.payload.result;
    var box = fit();
    var ctx = el.canvas.getContext('2d');
    var W = box.w;
    var S = state.span;
    var x0 = windowStart(t);
    var x1 = x0 + S;
    var analysedEnd = (r.seconds || []).length;
    var segs = r.flagged_segments || [];
    var spans = state.payload.review ? state.payload.review.spans : [];
    var k, row, top, i, a, b, X, step, tt, lab;

    function X_(sec) {
      return (sec - x0) / S * W;
    }

    ctx.setTransform(box.dpr, 0, 0, box.dpr, 0, 0);
    ctx.clearRect(0, 0, W, box.h);
    ctx.font = '10px ' + c.mono;
    ctx.textBaseline = 'top';

    for (k = 0; k < ROWS.length; k++) {
      row = ROWS[k];
      top = k * (ROW_H + ROW_GAP);
      ctx.fillStyle = c.plane;
      ctx.fillRect(0, top, W, ROW_H);

      // detected reading (red wash) and reviewer spans (amber band on top)
      for (i = 0; i < segs.length; i++) {
        if (segs[i].end_s > x0 && segs[i].start_s < x1) {
          a = X_(segs[i].start_s);
          b = X_(segs[i].end_s);
          ctx.fillStyle = 'rgba(208,59,59,.2)';
          ctx.fillRect(a, top, b - a, ROW_H);
        }
      }
      for (i = 0; i < spans.length; i++) {
        if (spans[i].end_s > x0 && spans[i].start_s < x1) {
          a = X_(spans[i].start_s);
          b = X_(spans[i].end_s);
          ctx.fillStyle = c.amber;
          ctx.fillRect(a, top, b - a, 3);
        }
      }
      // not analysed (decoding stopped before the end of the file)
      if (x1 > analysedEnd) {
        a = Math.max(0, X_(analysedEnd));
        ctx.fillStyle = 'rgba(255,255,255,.04)';
        ctx.fillRect(a, top, W - a, ROW_H);
      }

      if (row.key === 'prob') {
        drawProb(ctx, r, top, X_, x0, x1, W);
      } else {
        drawTrace(ctx, d[row.key], state.ranges[row.key], top, X_, x0, x1);
        if (row.blinks) {
          drawTicks(ctx, d.blink, top + ROW_H - 7, 6, c.ink2, X_, x0, x1);
        }
        if (row.speaking) {
          drawBand(ctx, d.speaking, top + ROW_H - 4, 4, c.good, X_, x0, x1);
        }
      }

      ctx.fillStyle = c.mute;
      ctx.fillText(row.label, LABEL_PAD, top + LABEL_PAD);
      ctx.strokeStyle = c.hairline;
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, top + 0.5, W - 1, ROW_H - 1);
    }

    // time axis
    top = ROWS.length * ROW_H + (ROWS.length - 1) * ROW_GAP;
    step = tickStep(S);
    ctx.fillStyle = c.mute;
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    for (tt = Math.ceil(x0 / step) * step; tt <= x1; tt += step) {
      X = X_(tt);
      ctx.fillRect(X, top, 1, 4);
      lab = app.format.mmss(tt);
      if (X > 18 && X < W - 18) {
        ctx.fillText(lab, X, top + 7);
      }
    }
    ctx.textAlign = 'left';

    // playhead
    X = X_(t);
    ctx.fillStyle = c.ink;
    ctx.fillRect(Math.round(X) - 1, 0, 2, top);
    state.drawnT = t;
    state.drawnW = W;
  }

  /** One channel as a line (gaps where the face was lost). */
  function drawTrace(ctx, xs, range, top, X_, x0, x1) {
    var d = state.data;
    var hz = d.hz;
    var i0 = Math.max(0, Math.floor((x0 - d.t0) * hz) - 1);
    var i1 = Math.min(xs.length - 1, Math.ceil((x1 - d.t0) * hz) + 1);
    var lo = range[0];
    var span = range[1] - range[0];
    var pen = false;
    var i, v, y;
    ctx.strokeStyle = state.colors.base;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (i = i0; i <= i1; i++) {
      v = xs[i];
      if (v === null || !isFinite(v)) {
        pen = false;
        continue;
      }
      y = top + ROW_H - 4 - Math.max(0, Math.min(1, (v - lo) / span)) * (ROW_H - 18);
      if (pen) {
        ctx.lineTo(X_(d.t0 + i / hz), y);
      } else {
        ctx.moveTo(X_(d.t0 + i / hz), y);
        pen = true;
      }
    }
    ctx.stroke();
  }

  /** Per-second reading probability with the per-second threshold. */
  function drawProb(ctx, r, top, X_, x0, x1, W) {
    var sec = r.seconds || [];
    var thr = r.second_threshold;
    var i0 = Math.max(0, Math.floor(x0) - 1);
    var i1 = Math.min(sec.length - 1, Math.ceil(x1) + 1);
    var pen = false;
    var i, y;
    function Y(p) {
      return top + ROW_H - 4 - Math.max(0, Math.min(1, p)) * (ROW_H - 18);
    }
    ctx.strokeStyle = state.colors.amber;
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Y(thr));
    ctx.lineTo(W, Y(thr));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = state.colors.flag;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (i = i0; i <= i1; i++) {
      if (sec[i] === null || sec[i] === undefined) {
        pen = false;
        continue;
      }
      y = Y(sec[i]);
      if (pen) {
        ctx.lineTo(X_(i + 0.5), y);
      } else {
        ctx.moveTo(X_(i + 0.5), y);
        pen = true;
      }
    }
    ctx.stroke();
  }

  /** Short vertical ticks wherever `xs` is on (blinks). */
  function drawTicks(ctx, xs, y, h, color, X_, x0, x1) {
    var d = state.data;
    var i0 = Math.max(0, Math.floor((x0 - d.t0) * d.hz));
    var i1 = Math.min(xs.length - 1, Math.ceil((x1 - d.t0) * d.hz));
    var i;
    ctx.fillStyle = color;
    for (i = i0; i <= i1; i++) {
      if (xs[i] && xs[i] >= 0.5) {
        ctx.fillRect(X_(d.t0 + i / d.hz), y, 1.5, h);
      }
    }
  }

  /** A band whose opacity follows `xs` (speaking). */
  function drawBand(ctx, xs, y, h, color, X_, x0, x1) {
    var d = state.data;
    var i0 = Math.max(0, Math.floor((x0 - d.t0) * d.hz));
    var i1 = Math.min(xs.length - 1, Math.ceil((x1 - d.t0) * d.hz));
    var w = Math.max(1, X_(1 / d.hz) - X_(0)) + 0.5;
    var i;
    ctx.fillStyle = color;
    for (i = i0; i <= i1; i++) {
      if (xs[i] && xs[i] >= 0.5) {
        ctx.fillRect(X_(d.t0 + i / d.hz), y, w, h);
      }
    }
  }

  /**
   * Called on every playback / seek update by review.js.
   * @param {number} t  seconds on the video clock
   * @param {number=} duration  the player's current duration
   */
  function update(t, duration) {
    var W;
    if (!state.data || el.fig.hidden) {
      return;
    }
    if (duration && duration > state.duration) {
      state.duration = duration;
    }
    state.t = t;
    W = el.plot.clientWidth;
    // skip frames that would move the playhead by less than half a pixel
    if (W === state.drawnW && Math.abs(t - state.drawnT) < 0.5 * state.span / Math.max(W, 1)) {
      return;
    }
    draw(t);
  }

  /**
   * Fetches the traces for a payload and draws them.
   * @param {Object} payload
   */
  function load(payload) {
    var token = ++state.token;
    state.payload = payload;
    state.data = null;
    el.fig.hidden = true;
    if (!payload.traces_url) {
      return;
    }
    app.http.getJson(payload.traces_url, function (err, data) {
      var key, i;
      if (err || token !== state.token || !data || !data.hz) {
        return;
      }
      state.data = data;
      state.colors = state.colors || colors();
      state.ranges = {};
      for (i = 0; i < ROWS.length; i++) {
        key = ROWS[i].key;
        if (key !== 'prob') {
          state.ranges[key] = robustRange(data[key] || []);
        }
      }
      state.duration = Math.max((payload.result.seconds || []).length,
        data.t0 + (data.eye_h || []).length / data.hz, payload.stats.video_seconds || 0);
      el.fig.hidden = false;
      draw(state.t);
    });
  }

  /** Clears the chart (a new job / candidate is loading). */
  function reset() {
    state.token += 1;
    state.payload = null;
    state.data = null;
    state.t = 0;
    state.drawnT = -1;
    el.fig.hidden = true;
  }

  /** Caches element references and binds the zoom chips, clicks and resizes. */
  function init() {
    el.fig = document.getElementById('traces-fig');
    el.plot = el.fig ? el.fig.querySelector('.traces__plot') : null;
    el.canvas = document.getElementById('traces-canvas');
    el.zoom = el.fig ? el.fig.querySelector('.traces__zoom') : null;
    if (!el.fig || !el.canvas) {
      return;
    }
    el.zoom.addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('[data-span]') : null;
      var chips, i;
      if (!btn) {
        return;
      }
      state.span = parseFloat(btn.getAttribute('data-span'));
      chips = el.zoom.querySelectorAll('[data-span]');
      for (i = 0; i < chips.length; i++) {
        chips[i].classList.toggle('is-on', chips[i] === btn);
        chips[i].setAttribute('aria-pressed', String(chips[i] === btn));
      }
      if (state.data) {
        draw(state.t);
      }
    });
    el.canvas.addEventListener('click', function (e) {
      var box = el.canvas.getBoundingClientRect();
      var t;
      if (!state.data) {
        return;
      }
      t = windowStart(state.t) + (e.clientX - box.left) / box.width * state.span;
      app.review.seekTo(t, false, false);
    });
    window.addEventListener('resize', function () {
      if (state.data && !el.fig.hidden) {
        draw(state.t);
      }
    });
  }

  app.traces = { init: init, load: load, update: update, reset: reset };
})(window.GazeApp = window.GazeApp || {});
