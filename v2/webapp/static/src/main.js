/* main.js (v2) — entry point. Two ways into the result view:
     * a pre-scored candidate from the library (#cand=<id>): the payload, video,
       traces and report are already on the server, so it opens instantly;
     * an upload: polls the job until the server finishes. The job id is kept in
       the URL hash (#job=…) so a reload picks the same job back up. */
(function (app) {
  'use strict';

  var POLL_MS = 1000;
  var MAX_POLL_FAILURES = 5;
  var JOB_HASH = /^#job=([0-9a-f]{16})$/;
  var CAND_HASH = /^#cand=([a-z0-9_-]{1,64})$/;
  var FOLDER_TEXT = {
    cheating_folder: 'cheating_folder · labelled cheating',
    not_cheating_folder: 'not_cheating_folder · labelled clean',
    new_not_cheating: 'new_not_cheating · labelled clean'
  };

  var el = {};
  var state = {
    info: null, jobId: null, uploadXhr: null, pollXhr: null,
    pollTimer: null, pollFailures: 0,
    file: null,   // the recording just uploaded — the review player plays it locally
    candId: null, candXhr: null
  };

  /**
   * Shows exactly one of the three panels.
   * @param {HTMLElement} panel
   */
  function show(panel) {
    el.panelLibrary.hidden = panel !== el.panelIntake;
    el.panelIntake.hidden = panel !== el.panelIntake;
    el.panelLive.hidden = panel !== el.panelLive;
    el.panelError.hidden = panel !== el.panelError;
  }

  /**
   * Full-page fault (the server can't be reached at all).
   * @param {string} msg
   */
  function fail(msg) {
    el.errorMsg.textContent = msg;
    show(el.panelError);
  }

  /**
   * Error inside the analysis panel (a single job failed).
   * @param {string} msg
   */
  function showLiveError(msg) {
    el.liveError.textContent = msg;
    el.liveError.hidden = !msg;
  }

  /**
   * Switches the action buttons between "running" and "finished".
   * @param {boolean} terminal
   * @param {boolean} hasResult
   */
  function setButtons(terminal, hasResult) {
    el.stop.hidden = terminal;
    el.again.hidden = !terminal;
    el.download.hidden = !(terminal && hasResult);
    el.segCsv.hidden = !(terminal && hasResult);
    el.report.hidden = !(terminal && hasResult);
  }

  /** Stops any in-flight polling. */
  function stopPolling() {
    if (state.pollTimer) {
      clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
    app.http.abort(state.pollXhr);
    state.pollXhr = null;
  }

  /**
   * Remembers the job in the URL so a reload resumes it.
   * @param {string|null} id
   */
  function setHash(id) {
    var url = window.location.pathname + window.location.search + (id ? '#job=' + id : '');
    window.history.replaceState(null, '', url);
  }

  /**
   * Handles a job that reached done / error / cancelled.
   * @param {Object} job
   */
  function onTerminal(job) {
    var base = '/api/jobs/' + job.id;
    stopPolling();
    if (job.status === 'done' && job.result) {
      el.download.href = base + '/result.json';
      el.segCsv.href = base + '/segments.csv';
      el.report.href = base + '/report';
      app.result.render(job.result, state.file);
      setButtons(true, true);
      return;
    }
    if (job.status === 'error') {
      showLiveError(job.error || 'The analysis failed.');
    }
    setButtons(true, false);
  }

  /** One poll of the job status; reschedules itself until the job ends. */
  function poll() {
    state.pollXhr = app.http.getJson('/api/jobs/' + state.jobId, function (err, job) {
      state.pollXhr = null;
      if (err) {
        state.pollFailures += 1;
        if (state.pollFailures >= MAX_POLL_FAILURES || /No such job/.test(err.message)) {
          showLiveError(err.message);
          setButtons(true, false);
          return;
        }
        state.pollTimer = setTimeout(poll, POLL_MS * 2);
        return;
      }
      state.pollFailures = 0;
      showLiveError(null);
      app.progress.renderJob(job);
      if (job.status === 'done' || job.status === 'error' || job.status === 'cancelled') {
        onTerminal(job);
        return;
      }
      state.pollTimer = setTimeout(poll, POLL_MS);
    });
  }

  /**
   * Starts following a server-side job.
   * @param {string} id
   */
  function follow(id) {
    state.jobId = id;
    state.pollFailures = 0;
    setHash(id);
    poll();
  }

  /**
   * Switches the analysis panel between a library candidate and an upload job.
   * @param {boolean} on
   */
  function setLibraryMode(on) {
    el.panelLive.classList.toggle('is-library', on);
    el.cand.hidden = !on;
    el.liveLabel.textContent = on ? '01 — Candidate' : '02 — Analysis';
    if (!on) {
      state.candId = null;
    }
  }

  /** Resets the analysis panel to its initial look. */
  function resetLive() {
    stopPolling();
    app.http.abort(state.candXhr);
    state.candXhr = null;
    setLibraryMode(false);
    app.result.hide();
    showLiveError(null);
    setButtons(false, false);
  }

  /**
   * Uploads the chosen file and starts following the new job.
   * @param {File} file
   * @param {number|null} maxMinutes
   */
  function submit(file, maxMinutes) {
    var form = new FormData();
    form.append('file', file, file.name);
    if (maxMinutes) {
      form.append('max_minutes', String(maxMinutes));
    }
    resetLive();
    state.file = file;
    show(el.panelLive);
    app.progress.start(state.info ? state.info.runtime.cnn : 'cpu');
    app.progress.renderUpload(0);

    state.uploadXhr = app.http.postForm('/api/jobs', form,
      function (frac) { app.progress.renderUpload(frac); },
      function (err, data) {
        state.uploadXhr = null;
        if (err) {
          showLiveError(err.message);
          setButtons(true, false);
          return;
        }
        follow(data.id);
      });
  }

  /** Cancel: abort the upload, or ask the server to drop the job. */
  function cancel() {
    if (state.uploadXhr) {
      app.http.abort(state.uploadXhr);
      state.uploadXhr = null;
      backToIntake();
      return;
    }
    if (state.jobId) {
      el.stop.disabled = true;
      app.http.del('/api/jobs/' + state.jobId, function (err) {
        el.stop.disabled = false;
        if (err) {
          showLiveError(err.message);
        }
      });
    }
  }

  /** Returns to the intake panel for another recording. */
  function backToIntake() {
    var last = state.candId;
    resetLive();
    state.jobId = null;
    state.file = null;
    setHash(null);
    app.intake.reset();
    show(el.panelIntake);
    if (last) {
      app.library.reveal(last);
    }
  }

  // ---------------------------------------------------------------- library

  /**
   * The candidate header: who, what the folder says, whether the model agrees.
   * @param {Object} payload
   */
  function renderCandidate(payload) {
    var f = app.format;
    var c = payload.candidate;
    var r = payload.result;
    var row = app.library.find(c.id);
    var nb = app.library.neighbours(c.id);
    var media = row ? row.media_seconds : null;
    var sh = c.shipped;
    el.candName.textContent = c.name;
    el.candIr.textContent = 'ir' + c.ir + (c.reviewed ? ' · manually reviewed' : '');
    el.candFolder.textContent = FOLDER_TEXT[c.folder] || c.folder;
    el.candCorrect.textContent = (c.correct ? '✓ agrees' : '✗ disagrees') + ' — model says ' +
      (r.verdict === 'CHEATING' ? 'reading' : 'clear');
    el.candCorrect.className = c.correct ? 'is-yes' : 'is-no';
    el.candScoring.textContent = 'out-of-fold model (never trained on this video)';
    el.candScoring.title = c.experiment;
    el.candShipped.textContent = (sh.verdict === 'CHEATING' ? 'reading' : 'clear') + ' · score ' +
      f.fixed(sh.video_stat_value, 3) + ' · reading ' + f.clock(sh.reading_seconds);
    el.candShipped.title = 'The shipped model trained on these recordings, so this is in-sample ' +
      '(optimistic); the out-of-fold verdict above is the honest one.';
    el.candPos.textContent = nb.total ? nb.pos + ' / ' + nb.total : '';
    el.candPrev.disabled = !nb.prev;
    el.candNext.disabled = !nb.next;
    el.candPrev.setAttribute('data-id', nb.prev || '');
    el.candNext.setAttribute('data-id', nb.next || '');
    el.candCoverage.hidden = !(media && media - payload.stats.video_seconds > 3);
    if (!el.candCoverage.hidden) {
      el.candCoverage.textContent = 'Only the first ' + f.clock(payload.stats.video_seconds) + ' of ' +
        f.clock(media) + ' was analysed (the face tracker stopped decoding there); the rest of the ' +
        'timeline is hatched and carries no result.';
    }
  }

  /**
   * Opens a pre-scored candidate (called by the router).
   * @param {string} id
   */
  function openCandidate(id) {
    if (state.candId === id && !el.panelLive.hidden) {
      return;
    }
    resetLive();
    state.jobId = null;
    state.file = null;
    setLibraryMode(true);
    state.candId = id;
    show(el.panelLive);
    el.liveLabel.scrollIntoView({ block: 'start' });
    el.candName.textContent = 'loading…';
    state.candXhr = app.http.getJson('/api/library/' + id, function (err, payload) {
      var base = '/api/library/' + id;
      state.candXhr = null;
      if (state.candId !== id) {
        return;
      }
      if (err) {
        showLiveError(err.message);
        return;
      }
      el.candJson.href = base + '/result.json';
      el.candCsv.href = base + '/segments.csv';
      el.candReport.href = base + '/report';
      setButtons(true, false);
      app.result.render(payload, null);
      renderCandidate(payload);
    });
  }

  /**
   * Navigates to a candidate (pushes history, so Back returns to the list).
   * @param {string} id
   */
  function goCandidate(id) {
    if (id) {
      window.location.hash = 'cand=' + id;
    }
  }

  /** Follows the URL hash: a candidate, a job, or the landing page. */
  function route() {
    var m = CAND_HASH.exec(window.location.hash);
    if (m) {
      openCandidate(m[1]);
      return;
    }
    if (state.candId) {
      backToIntake();
    }
  }

  /**
   * Fills the masthead spec bar from GET /api/info.
   * @param {Object} info
   */
  function renderSpecs(info) {
    var m = info.model;
    var rt = info.runtime;
    el.specKind.textContent = m.kind + (m.experiment ? ' · ' + m.experiment : '');
    el.specThreshold.textContent = m.second_threshold.toFixed(3);
    el.specRule.textContent = m.video_stat === 'reading_seconds'
      ? 'reading ≥ ' + app.format.clock(m.video_threshold)
      : (m.video_stat === 'speech_prob_mean' ? 'avg P(reading) while speaking' : m.video_stat) +
        ' ≥ ' + m.video_threshold.toFixed(2);
    el.specWindow.textContent = m.window_seconds + 's / ' + m.stride_seconds + 's';
    el.specMediapipe.textContent = rt.mediapipe + ' × ' + rt.mediapipe_threads + ' threads';
    el.specMediapipe.title = rt.mediapipe_threads + ' of ' + rt.cpu_count + ' cores';
    el.specCnn.textContent = rt.cnn + (rt.workers > 1 ? ' · ' + rt.workers + ' workers' : '');
    el.specCnn.title = rt.cpu;
  }

  /** Loads the server info, then resumes a job from the URL hash if present. */
  function boot() {
    app.http.getJson('/api/info', function (err, info) {
      var match;
      if (err) {
        el.specCnn.textContent = 'offline';
        fail('Could not reach the analysis server: ' + err.message +
          ' Start it with: venv/bin/python v2/webapp/server.py');
        return;
      }
      state.info = info;
      renderSpecs(info);
      app.intake.setReady(info.accepted);

      // the library first, so a #cand= link gets its list position and coverage
      app.library.load(function () {
        if (CAND_HASH.test(window.location.hash)) {
          route();
        }
      });

      match = JOB_HASH.exec(window.location.hash);
      if (match) {
        resetLive();
        show(el.panelLive);
        app.progress.start(info.runtime.cnn);
        follow(match[1]);
      }
    });
  }

  /** Entry point. */
  function init() {
    el.panelLibrary = document.getElementById('panel-library');
    el.panelIntake = document.getElementById('panel-intake');
    el.liveLabel = document.getElementById('live-h');
    el.cand = document.getElementById('cand');
    el.candName = document.getElementById('cand-name');
    el.candIr = document.getElementById('cand-ir');
    el.candFolder = document.getElementById('cand-folder');
    el.candCorrect = document.getElementById('cand-correct');
    el.candScoring = document.getElementById('cand-scoring');
    el.candShipped = document.getElementById('cand-shipped');
    el.candCoverage = document.getElementById('cand-coverage');
    el.candPos = document.getElementById('cand-pos');
    el.candPrev = document.getElementById('cand-prev');
    el.candNext = document.getElementById('cand-next');
    el.candBack = document.getElementById('cand-back');
    el.candJson = document.getElementById('cand-json');
    el.candCsv = document.getElementById('cand-csv');
    el.candReport = document.getElementById('cand-report');
    el.panelLive = document.getElementById('panel-live');
    el.panelError = document.getElementById('panel-error');
    el.errorMsg = document.getElementById('error-msg');
    el.liveError = document.getElementById('live-error');
    el.stop = document.getElementById('live-stop');
    el.again = document.getElementById('live-again');
    el.download = document.getElementById('download');
    el.segCsv = document.getElementById('segments-csv');
    el.report = document.getElementById('report-link');
    el.retry = document.getElementById('retry');
    el.specKind = document.getElementById('spec-kind');
    el.specThreshold = document.getElementById('spec-threshold');
    el.specRule = document.getElementById('spec-rule');
    el.specWindow = document.getElementById('spec-window');
    el.specMediapipe = document.getElementById('spec-mediapipe');
    el.specCnn = document.getElementById('spec-cnn');

    app.intake.init({ onSubmit: submit });
    app.library.init({ onOpen: goCandidate });
    app.traces.init();
    app.progress.init();
    app.review.init();
    app.compare.init();
    app.result.init();

    el.stop.addEventListener('click', cancel);
    el.again.addEventListener('click', backToIntake);
    el.retry.addEventListener('click', function () { window.location.reload(); });
    el.candBack.addEventListener('click', backToIntake);
    el.candPrev.addEventListener('click', function () { goCandidate(el.candPrev.getAttribute('data-id')); });
    el.candNext.addEventListener('click', function () { goCandidate(el.candNext.getAttribute('data-id')); });
    window.addEventListener('hashchange', route);

    boot();
  }

  document.addEventListener('DOMContentLoaded', init);
})(window.GazeApp = window.GazeApp || {});
