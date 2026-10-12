/* review.js — plays the analysed recording with the suspicious parts marked.
   A heat strip under the video shows every window's score on the video's own
   clock (red = flagged); flagged segments are jump buttons; while playback is
   inside a flagged segment the screen glows red and names the reason.
   The video plays from the user's own file via an object URL — it is never
   uploaded again — or from the server when it was started with --keep-uploads. */
(function (app) {
  'use strict';

  var SEEK_STEP = 5;          // arrow keys on the strip, seconds
  var SEEK_PAGE = 30;         // page up / down, seconds
  var SVG_W = 1000;
  var SVG_H = 40;

  var el = {};
  var state = {
    payload: null, windows: [], segments: [], cells: [],
    t0: 0, analysedEnd: 0, duration: 0, flagThr: 1,
    objectUrl: null, cursor: 0, hasVideo: false, raf: 0, currentSeg: -2
  };

  /**
   * Same reduction the server applies to the uploaded filename for display, so a
   * re-attached file can be compared with the analysed one.
   * @param {string} name
   * @returns {string}
   */
  function displayName(name) {
    return (name || 'video').replace(/[^A-Za-z0-9 ._()\-]/g, '_').slice(0, 120);
  }

  /**
   * Splits the timeline at every window boundary and scores each piece with the
   * highest-scoring window covering it (windows overlap: 20 s long, 10 s apart).
   * @param {Array<Object>} windows  [{start_s, end_s, prob}] on the analysis clock
   * @param {number} t0  video time of the first analysed frame
   * @returns {Array<Object>} [{start, end, prob, idx}] on the video clock
   */
  function buildCells(windows, t0) {
    var bounds = [], uniq = [], cells = [];
    var i, j, a, b, w, bestIdx;
    for (i = 0; i < windows.length; i++) {
      bounds.push(windows[i].start_s, windows[i].end_s);
    }
    bounds.sort(function (x, y) { return x - y; });
    for (i = 0; i < bounds.length; i++) {
      if (!uniq.length || bounds[i] - uniq[uniq.length - 1] > 1e-6) {
        uniq.push(bounds[i]);
      }
    }
    for (i = 0; i < uniq.length - 1; i++) {
      a = uniq[i];
      b = uniq[i + 1];
      bestIdx = -1;
      for (j = 0; j < windows.length; j++) {
        w = windows[j];
        if (w.start_s <= a + 1e-6 && w.end_s >= b - 1e-6 &&
            (bestIdx < 0 || w.prob > windows[bestIdx].prob)) {
          bestIdx = j;
        }
      }
      if (bestIdx >= 0) {
        cells.push({ start: t0 + a, end: t0 + b, prob: windows[bestIdx].prob, idx: bestIdx });
      }
    }
    return cells;
  }

  /**
   * The highest-scoring window that contains analysis time `a`, or -1.
   * @param {number} a  seconds on the analysis clock
   * @returns {number}
   */
  function windowAt(a) {
    var best = -1, i, w;
    for (i = 0; i < state.windows.length; i++) {
      w = state.windows[i];
      if (w.start_s <= a && a < w.end_s && (best < 0 || w.prob > state.windows[best].prob)) {
        best = i;
      }
    }
    return best;
  }

  /**
   * Index of the flagged segment containing analysis time `a`, or -1.
   * @param {number} a
   * @returns {number}
   */
  function segmentAt(a) {
    var i, s;
    for (i = 0; i < state.segments.length; i++) {
      s = state.segments[i];
      if (s.start_s <= a && a < s.end_s) {
        return i;
      }
    }
    return -1;
  }

  /** Draws the heat strip for the current duration. */
  function renderStrip() {
    var D = state.duration || 1;
    var parts = [];
    var i, c, x, w, hit, x0;
    parts.push('<svg viewBox="0 0 ' + SVG_W + ' ' + SVG_H + '" preserveAspectRatio="none" aria-hidden="true">');
    parts.push('<defs><pattern id="strip-hatch" width="8" height="8" patternUnits="userSpaceOnUse" ' +
      'patternTransform="rotate(45)"><rect width="8" height="8" fill="var(--plane)"/>' +
      '<rect width="3" height="8" fill="var(--hairline)"/></pattern></defs>');
    parts.push('<rect x="0" y="0" width="' + SVG_W + '" height="' + SVG_H + '" fill="var(--plane)"/>');
    for (i = 0; i < state.cells.length; i++) {
      c = state.cells[i];
      hit = c.prob >= state.flagThr;
      x = (c.start / D) * SVG_W;
      w = Math.max(0.6, ((c.end - c.start) / D) * SVG_W);
      parts.push('<rect x="' + x.toFixed(2) + '" y="0" width="' + w.toFixed(2) + '" height="' + SVG_H + '" ' +
        (hit ? 'fill="var(--series-flag)"'
             : 'fill="var(--series-base)" fill-opacity="' + (0.1 + 0.8 * c.prob).toFixed(3) + '"') + '/>');
    }
    if (state.duration > state.analysedEnd + 0.5) {
      x0 = (state.analysedEnd / D) * SVG_W;
      parts.push('<rect x="' + x0.toFixed(2) + '" y="0" width="' + (SVG_W - x0).toFixed(2) +
        '" height="' + SVG_H + '" fill="url(#strip-hatch)"/>');
    }
    parts.push('</svg>');
    el.cells.innerHTML = parts.join('');
    el.stripEnd.textContent = app.format.mmss(D);
    el.strip.setAttribute('aria-valuemax', String(Math.round(D)));
  }

  /** Renders the flagged segments as jump buttons. */
  function renderSegments() {
    var f = app.format;
    var i, s, li, btn, range, reason, peak;
    el.segments.textContent = '';
    el.segmentsEmpty.hidden = state.segments.length > 0;
    for (i = 0; i < state.segments.length; i++) {
      s = state.segments[i];
      li = document.createElement('li');
      btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'seg seg--jump';
      btn.setAttribute('data-seg', String(i));
      btn.setAttribute('aria-label', 'Play flagged segment ' + f.mmss(state.t0 + s.start_s) + ' to ' +
        f.mmss(state.t0 + s.end_s) + ', ' + s.reason);
      range = document.createElement('span');
      range.className = 'seg__range';
      range.textContent = '▶ ' + f.mmss(state.t0 + s.start_s) + '–' + f.mmss(state.t0 + s.end_s);
      reason = document.createElement('span');
      reason.className = 'seg__reason';
      reason.textContent = s.reason;
      peak = document.createElement('span');
      peak.className = 'seg__peak';
      peak.textContent = 'peak ' + s.peak_prob.toFixed(3);
      btn.appendChild(range);
      btn.appendChild(reason);
      btn.appendChild(peak);
      li.appendChild(btn);
      el.segments.appendChild(li);
    }
  }

  /**
   * Current playback (or scrub) position in seconds on the video clock.
   * @returns {number}
   */
  function currentTime() {
    return state.hasVideo ? el.video.currentTime : state.cursor;
  }

  /** Updates the playhead, the "Now" readout and the suspicious-state styling. */
  function updateNow() {
    var f = app.format;
    var t = currentTime();
    var a = t - state.t0;
    var wi = windowAt(a);
    var si = segmentAt(a);
    var w = wi >= 0 ? state.windows[wi] : null;
    var buttons, i;

    el.head.style.left = (Math.min(1, Math.max(0, t / (state.duration || 1))) * 100) + '%';
    el.strip.setAttribute('aria-valuenow', String(Math.round(t)));
    el.strip.setAttribute('aria-valuetext', f.mmss(t));
    el.nowTime.textContent = f.mmss(t);
    el.nowScore.textContent = w ? w.prob.toFixed(3) : (a > state.analysedEnd - state.t0 ? 'not analysed' : '—');
    el.nowScore.className = w && w.prob >= state.flagThr ? 'is-flagged' : '';
    el.nowReasons.textContent = w && w.reasons.length ? w.reasons.join(', ') : '—';

    if (si === state.currentSeg) {
      return;
    }
    state.currentSeg = si;
    el.screen.classList.toggle('is-suspicious', si >= 0);
    el.flag.hidden = si < 0;
    if (si >= 0) {
      el.flag.textContent = 'Suspicious · ' + state.segments[si].reason;
    }
    buttons = el.segments.querySelectorAll('.seg--jump');
    for (i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle('is-current', i === si);
    }
  }

  /** Keeps the playhead smooth while the video plays (timeupdate is only ~4 Hz). */
  function tick() {
    updateNow();
    state.raf = window.requestAnimationFrame(tick);
  }

  /** Stops the animation loop. */
  function stopTick() {
    if (state.raf) {
      window.cancelAnimationFrame(state.raf);
      state.raf = 0;
    }
  }

  /**
   * Seeks the video (or just the timeline cursor when no video is loaded).
   * @param {number} t  seconds on the video clock
   * @param {boolean} play  start playback after seeking
   * @param {boolean} reveal  scroll the player into view
   */
  function seekTo(t, play, reveal) {
    var p;
    t = Math.max(0, Math.min(t, state.duration || t));
    if (reveal && el.review.getBoundingClientRect().top < 0) {
      el.review.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    if (state.hasVideo) {
      el.video.currentTime = t;
      if (play) {
        p = el.video.play();
        if (p && typeof p.catch === 'function') {
          p.catch(function () { /* autoplay refused: the seek still happened */ });
        }
      }
    } else {
      state.cursor = t;
    }
    updateNow();
  }

  /**
   * Plays window `i` (used by the chart bars and the window table).
   * @param {number} i
   */
  function seekWindow(i) {
    var w = state.windows[i];
    if (w) {
      seekTo(state.t0 + w.start_s, true, true);
    }
  }

  /**
   * Shows the "load the recording" panel in place of the video.
   * @param {string} msg
   */
  function showEmpty(msg) {
    state.hasVideo = false;
    el.video.hidden = true;
    el.empty.hidden = false;
    el.emptyMsg.textContent = msg;
  }

  /**
   * Points the player at a URL (object URL or server route).
   * @param {string} url
   */
  function setSource(url) {
    el.empty.hidden = true;
    el.video.hidden = false;
    state.hasVideo = true;
    el.video.src = url;
    el.video.load();
  }

  /**
   * Plays a local file, warning when it isn't obviously the analysed recording.
   * @param {File} file
   */
  function setFile(file) {
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
    }
    state.objectUrl = URL.createObjectURL(file);
    setSource(state.objectUrl);
    if (state.payload && displayName(file.name) !== state.payload.source) {
      el.note.textContent = 'This file is named "' + file.name + '", but the analysed one was "' +
        state.payload.source + '" — the marks only line up on the same recording.';
      el.note.hidden = false;
    } else {
      el.note.hidden = true;
    }
  }

  /**
   * Converts a pointer position on the strip to video time.
   * @param {MouseEvent} e
   * @returns {number}
   */
  function stripTime(e) {
    var box = el.strip.getBoundingClientRect();
    var frac = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width));
    return frac * (state.duration || 0);
  }

  /** Wires the player, strip, picker and jump buttons (once). */
  function bind() {
    el.video.addEventListener('timeupdate', updateNow);
    el.video.addEventListener('seeked', updateNow);
    el.video.addEventListener('play', function () { stopTick(); tick(); });
    el.video.addEventListener('pause', stopTick);
    el.video.addEventListener('ended', stopTick);
    el.video.addEventListener('loadedmetadata', function () {
      var d = el.video.duration;
      // Only trust a longer, finite container duration (it reveals an un-analysed
      // tail after "analyse first N minutes"); webm often reports Infinity.
      if (isFinite(d) && d > state.analysedEnd && d < state.analysedEnd * 50 + 3600) {
        state.duration = d;
        renderStrip();
        updateNow();
      }
    });
    el.video.addEventListener('error', function () {
      var src = el.video.getAttribute('src');
      if (!src) {
        return;
      }
      showEmpty('This browser can\'t play this file. The timeline and segments still work — ' +
        'open the recording in a desktop player at the times shown, or choose an mp4/webm copy.');
    });

    el.strip.addEventListener('click', function (e) { seekTo(stripTime(e), false, false); });
    el.strip.addEventListener('mousemove', function (e) {
      var t = stripTime(e);
      var a = t - state.t0;
      var wi = windowAt(a);
      var box = el.strip.getBoundingClientRect();
      el.tip.textContent = app.format.mmss(t) + (wi >= 0 ? ' · ' + state.windows[wi].prob.toFixed(3) : ' · not analysed');
      el.tip.style.left = (e.clientX - box.left) + 'px';
      el.tip.hidden = false;
    });
    el.strip.addEventListener('mouseleave', function () { el.tip.hidden = true; });
    el.strip.addEventListener('keydown', function (e) {
      var t = currentTime();
      var handled = true;
      if (e.key === 'ArrowLeft') { t -= SEEK_STEP; }
      else if (e.key === 'ArrowRight') { t += SEEK_STEP; }
      else if (e.key === 'PageDown') { t -= SEEK_PAGE; }
      else if (e.key === 'PageUp') { t += SEEK_PAGE; }
      else if (e.key === 'Home') { t = 0; }
      else if (e.key === 'End') { t = state.duration; }
      else { handled = false; }
      if (handled) {
        e.preventDefault();
        seekTo(t, false, false);
      }
    });

    el.segments.addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('.seg--jump') : null;
      var s;
      if (btn) {
        s = state.segments[parseInt(btn.getAttribute('data-seg'), 10)];
        seekTo(state.t0 + s.start_s, true, false);
      }
    });

    el.pick.addEventListener('click', function () { el.file.click(); });
    el.file.addEventListener('change', function () {
      if (el.file.files[0]) {
        setFile(el.file.files[0]);
      }
    });
  }

  /**
   * Loads a finished job into the reviewer.
   * @param {Object} payload  the server's result payload
   * @param {File|null} file  the recording the user uploaded, when still in memory
   */
  function load(payload, file) {
    var stats = payload.stats;
    var last;
    reset();
    state.payload = payload;
    state.windows = payload.windows;
    state.segments = payload.result.flagged_segments;
    state.flagThr = payload.result.flag_threshold;
    state.t0 = stats.t0 || 0;
    last = state.windows[state.windows.length - 1];
    state.analysedEnd = Math.max(state.t0 + (last ? last.end_s : 0), stats.video_seconds || 0);
    state.duration = state.analysedEnd;
    state.cells = buildCells(state.windows, state.t0);

    renderStrip();
    renderSegments();

    if (file) {
      setFile(file);
    } else if (payload.video_url) {
      setSource(payload.video_url);
    } else {
      showEmpty('Load "' + payload.source + '" to review it here.');
    }
    updateNow();
  }

  /** Releases the video and clears the reviewer. */
  function reset() {
    stopTick();
    el.video.pause();
    el.video.removeAttribute('src');
    el.video.load();
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    state.hasVideo = false;
    state.cursor = 0;
    state.currentSeg = -2;
    el.note.hidden = true;
    el.flag.hidden = true;
    el.screen.classList.remove('is-suspicious');
    el.file.value = '';
  }

  /** Caches element references and binds events. */
  function init() {
    var ids = {
      review: 'review', screen: 'review-screen', video: 'review-video', flag: 'review-flag',
      empty: 'review-empty', emptyMsg: 'review-empty-msg', pick: 'review-pick', file: 'review-file',
      strip: 'strip', cells: 'strip-cells', head: 'strip-head', tip: 'strip-tip', stripEnd: 'strip-end',
      note: 'review-note', nowTime: 'now-time', nowScore: 'now-score', nowReasons: 'now-reasons',
      segments: 'segments', segmentsEmpty: 'segments-empty'
    };
    var key;
    for (key in ids) {
      if (Object.prototype.hasOwnProperty.call(ids, key)) {
        el[key] = document.getElementById(ids[key]);
      }
    }
    if (el.video && el.strip) {
      bind();
    }
  }

  app.review = { init: init, load: load, reset: reset, seekWindow: seekWindow, seekTo: seekTo };
})(window.GazeApp = window.GazeApp || {});
