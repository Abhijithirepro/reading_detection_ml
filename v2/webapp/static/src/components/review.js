/* review.js (v2) — plays the analysed recording with the reading parts marked.
   The strip under the video has three lanes on the video's own clock:
     reading   per-second reading probability (brighter = more likely, red =
               inside a detected reading segment, hatched = not analysed)
     speaking  lip/jaw motion (the candidate is talking)
     reviewer  the manual review's reading spans, when one is attached
   Reading segments are jump buttons; while playback is inside one the screen
   glows red and names the signals. The video plays from the user's own file via
   an object URL — it is never uploaded again — or from the server when it was
   started with --keep-uploads. */
(function (app) {
  'use strict';

  var SEEK_STEP = 5;          // arrow keys on the strip, seconds
  var SEEK_PAGE = 30;         // page up / down, seconds
  var SVG_W = 1000;
  var SVG_H = 40;
  var LANE_READ = [0, 26];    // y, height of each lane inside the 40-unit strip
  var LANE_SPEAK = [28, 5];
  var LANE_REVIEW = [35, 5];

  var el = {};
  var state = {
    payload: null, windows: [], segments: [], seconds: [], speaking: [], review: null,
    analysedEnd: 0, duration: 0, flagThr: 1,
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
   * Index of the 5 s bin (chart / table row) containing video time `t`, or -1.
   * @param {number} t
   * @returns {number}
   */
  function windowAt(t) {
    var i, w;
    for (i = 0; i < state.windows.length; i++) {
      w = state.windows[i];
      if (w.start_s <= t && t < w.end_s) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Index of the reading segment containing video time `t`, or -1.
   * @param {number} t
   * @returns {number}
   */
  function segmentAt(t) {
    var i, s;
    for (i = 0; i < state.segments.length; i++) {
      s = state.segments[i];
      if (s.start_s <= t && t < s.end_s) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Rect markup for one lane cell.
   * @returns {string}
   */
  function rect(x0, x1, lane, attrs) {
    var D = state.duration || 1;
    var x = (x0 / D) * SVG_W;
    var w = Math.max(0.6, ((x1 - x0) / D) * SVG_W);
    return '<rect x="' + x.toFixed(2) + '" y="' + lane[0] + '" width="' + w.toFixed(2) +
      '" height="' + lane[1] + '" ' + attrs + '/>';
  }

  /** Draws the three-lane strip for the current duration. */
  function renderStrip() {
    var D = state.duration || 1;
    var parts = [];
    var i, p, x0, inSeg, sp, s;
    parts.push('<svg viewBox="0 0 ' + SVG_W + ' ' + SVG_H + '" preserveAspectRatio="none" aria-hidden="true">');
    parts.push('<defs><pattern id="strip-hatch" width="8" height="8" patternUnits="userSpaceOnUse" ' +
      'patternTransform="rotate(45)"><rect width="8" height="8" fill="var(--plane)"/>' +
      '<rect width="3" height="8" fill="var(--hairline)"/></pattern></defs>');
    parts.push('<rect x="0" y="0" width="' + SVG_W + '" height="' + SVG_H + '" fill="var(--plane)"/>');

    for (i = 0; i < state.seconds.length; i++) {
      p = state.seconds[i];
      if (p === null || p === undefined) {
        continue;
      }
      inSeg = segmentAt(i + 0.5) >= 0;
      parts.push(rect(i, i + 1, LANE_READ, inSeg ? 'fill="var(--series-flag)"'
        : 'fill="var(--series-base)" fill-opacity="' + (0.08 + 0.85 * p).toFixed(3) + '"'));
      sp = state.speaking[i] || 0;
      if (sp > 0.05) {
        parts.push(rect(i, i + 1, LANE_SPEAK, 'fill="var(--good)" fill-opacity="' + (0.2 + 0.8 * sp).toFixed(3) + '"'));
      }
    }
    if (state.review) {
      for (i = 0; i < state.review.spans.length; i++) {
        s = state.review.spans[i];
        parts.push(rect(s.start_s, s.end_s, LANE_REVIEW, 'fill="var(--amber)"'));
      }
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

  /** Renders the reading segments as jump buttons. */
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
      btn.setAttribute('aria-label', 'Play reading segment ' + f.mmss(s.start_s) + ' to ' +
        f.mmss(s.end_s) + ', ' + s.reason);
      range = document.createElement('span');
      range.className = 'seg__range';
      range.textContent = '▶ ' + f.mmss(s.start_s) + '–' + f.mmss(s.end_s);
      reason = document.createElement('span');
      reason.className = 'seg__reason';
      reason.textContent = s.reason;
      peak = document.createElement('span');
      peak.className = 'seg__peak';
      peak.textContent = Math.round(s.end_s - s.start_s) + 's · peak ' + s.peak_prob.toFixed(2);
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

  /** Updates the playhead, the "Now" readout and the reading-state styling. */
  function updateNow() {
    var f = app.format;
    var t = currentTime();
    var sec = Math.floor(t);
    var p = state.seconds[sec];
    var sp = state.speaking[sec];
    var si = segmentAt(t);
    var buttons, i;

    el.head.style.left = (Math.min(1, Math.max(0, t / (state.duration || 1))) * 100) + '%';
    el.strip.setAttribute('aria-valuenow', String(Math.round(t)));
    el.strip.setAttribute('aria-valuetext', f.mmss(t));
    el.nowTime.textContent = f.mmss(t);
    el.nowScore.textContent = (p === null || p === undefined) ? (t >= state.analysedEnd ? 'not analysed' : '—')
      : p.toFixed(3);
    el.nowScore.className = si >= 0 ? 'is-flagged' : '';
    if (el.nowSpeak) {
      el.nowSpeak.textContent = sp === undefined ? '—' : (sp >= 0.5 ? 'yes' : 'no') + ' (' + f.percent(sp) + ')';
    }
    el.nowReasons.textContent = si >= 0 && state.segments[si].reasons.length
      ? state.segments[si].reasons.join(', ') : '—';

    if (si === state.currentSeg) {
      return;
    }
    state.currentSeg = si;
    el.screen.classList.toggle('is-suspicious', si >= 0);
    el.flag.hidden = si < 0;
    if (si >= 0) {
      el.flag.textContent = 'Reading · ' + state.segments[si].reason;
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
   * Plays bin `i` (used by the chart bars and the table).
   * @param {number} i
   */
  function seekWindow(i) {
    var w = state.windows[i];
    if (w) {
      seekTo(w.start_s, true, true);
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
      var p = state.seconds[Math.floor(t)];
      var box = el.strip.getBoundingClientRect();
      el.tip.textContent = app.format.mmss(t) + (p === null || p === undefined
        ? ' · not analysed' : ' · reading ' + p.toFixed(2));
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
        seekTo(s.start_s, true, false);
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
   * Attaches (or replaces) the manual review drawn in the reviewer lane.
   * @param {Object|null} review  payload.review
   */
  function setReview(review) {
    state.review = review || null;
    renderStrip();
  }

  /**
   * Loads a finished job into the reviewer.
   * @param {Object} payload  the server's result payload
   * @param {File|null} file  the recording the user uploaded, when still in memory
   */
  function load(payload, file) {
    var r = payload.result;
    reset();
    state.payload = payload;
    state.windows = payload.windows;
    state.segments = r.flagged_segments;
    state.seconds = r.seconds || [];
    state.speaking = r.speaking || [];
    state.review = payload.review || null;
    state.flagThr = r.flag_threshold;
    state.analysedEnd = Math.max(state.seconds.length, payload.stats.video_seconds || 0);
    state.duration = state.analysedEnd;

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
      note: 'review-note', nowTime: 'now-time', nowScore: 'now-score', nowSpeak: 'now-speak',
      nowReasons: 'now-reasons', segments: 'segments', segmentsEmpty: 'segments-empty'
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

  app.review = {
    init: init, load: load, reset: reset, seekWindow: seekWindow, seekTo: seekTo, setReview: setReview
  };
})(window.GazeApp = window.GazeApp || {});
