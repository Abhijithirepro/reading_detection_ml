/* main.js — entry point. Uploads a recording, polls the job until the server
   finishes, then hands the payload to the result view. The job id is kept in the
   URL hash (#job=…) so a reload picks the same job back up. */
(function (app) {
  'use strict';

  var POLL_MS = 1000;
  var MAX_POLL_FAILURES = 5;
  var JOB_HASH = /^#job=([0-9a-f]{16})$/;

  var el = {};
  var state = {
    info: null, jobId: null, uploadXhr: null, pollXhr: null,
    pollTimer: null, pollFailures: 0,
    file: null   // the recording just uploaded — the review player plays it locally
  };

  /**
   * Shows exactly one of the three panels.
   * @param {HTMLElement} panel
   */
  function show(panel) {
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

  /** Resets the analysis panel to its initial look. */
  function resetLive() {
    stopPolling();
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
    resetLive();
    state.jobId = null;
    state.file = null;
    setHash(null);
    app.intake.reset();
    show(el.panelIntake);
  }

  /**
   * Fills the masthead spec bar from GET /api/info.
   * @param {Object} info
   */
  function renderSpecs(info) {
    var m = info.model;
    var rt = info.runtime;
    el.specKind.textContent = m.kind;
    el.specThreshold.textContent = m.threshold.toFixed(3);
    el.specWindow.textContent = m.window_seconds + 's / ' + m.stride_seconds + 's';
    el.specMediapipe.textContent = rt.mediapipe + ' × ' + rt.mediapipe_threads + ' threads';
    el.specMediapipe.title = rt.mediapipe_threads + ' of ' + rt.cpu_count + ' cores';
    el.specCnn.textContent = rt.cnn + (rt.workers > 1 ? ' · ' + rt.workers + ' workers' : '');
    el.specCnn.title = rt.cpu + ' · torch ' + rt.torch;
  }

  /** Loads the server info, then resumes a job from the URL hash if present. */
  function boot() {
    app.http.getJson('/api/info', function (err, info) {
      var match;
      if (err) {
        el.specCnn.textContent = 'offline';
        fail('Could not reach the analysis server: ' + err.message +
          ' Start it with: venv/bin/python webapp/server.py');
        return;
      }
      state.info = info;
      renderSpecs(info);
      app.intake.setReady(info.accepted);

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
    el.panelIntake = document.getElementById('panel-intake');
    el.panelLive = document.getElementById('panel-live');
    el.panelError = document.getElementById('panel-error');
    el.errorMsg = document.getElementById('error-msg');
    el.liveError = document.getElementById('live-error');
    el.stop = document.getElementById('live-stop');
    el.again = document.getElementById('live-again');
    el.download = document.getElementById('download');
    el.report = document.getElementById('report-link');
    el.retry = document.getElementById('retry');
    el.specKind = document.getElementById('spec-kind');
    el.specThreshold = document.getElementById('spec-threshold');
    el.specWindow = document.getElementById('spec-window');
    el.specMediapipe = document.getElementById('spec-mediapipe');
    el.specCnn = document.getElementById('spec-cnn');

    app.intake.init({ onSubmit: submit });
    app.progress.init();
    app.review.init();
    app.result.init();

    el.stop.addEventListener('click', cancel);
    el.again.addEventListener('click', backToIntake);
    el.retry.addEventListener('click', function () { window.location.reload(); });

    boot();
  }

  document.addEventListener('DOMContentLoaded', init);
})(window.GazeApp = window.GazeApp || {});
