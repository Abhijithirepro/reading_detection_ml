/* progress.js — the pipeline ladder, status badge, readouts and meter while a
   job uploads, waits in the queue and runs on the server. */
(function (app) {
  'use strict';

  var STEPS = ['upload', 'queued', 'frames', 'windows', 'report'];
  var el = {};
  var timing = { submittedAt: 0, runStartedAt: 0 };
  var cnnDevice = 'cpu';

  /**
   * Status line for each backend stage.
   * @param {string} stage
   * @param {Object} job
   * @returns {string}
   */
  function stageText(stage, job) {
    switch (stage) {
      case 'queued':
        return 'waiting for a free worker — position ' + (job.queue_position || 1);
      case 'frames':
        return 'tracking face + iris with MediaPipe on the CPU';
      case 'windows':
        return 'building 20 s windows and scoring them with the CNN (' + cnnDevice + ')';
      case 'report':
        return 'rendering the detailed report';
      default:
        return '';
    }
  }

  /**
   * Lights the ladder up to `stage`.
   * @param {string} stage   current step key
   * @param {boolean} finished all steps complete
   * @param {boolean} failed  current step failed
   */
  function paintLadder(stage, finished, failed) {
    var idx = STEPS.indexOf(stage);
    var items = el.ladder.querySelectorAll('.ladder__step');
    var i, step, pos;
    for (i = 0; i < items.length; i++) {
      step = items[i];
      pos = STEPS.indexOf(step.getAttribute('data-step'));
      step.classList.remove('is-done', 'is-active', 'is-failed');
      if (finished || pos < idx) {
        step.classList.add('is-done');
      } else if (pos === idx) {
        step.classList.add(failed ? 'is-failed' : 'is-active');
      }
    }
  }

  /**
   * Sets the meter to a fraction, or to an indeterminate sweep when null.
   * @param {number|null} frac
   */
  function setMeter(frac) {
    var indeterminate = frac === null;
    el.meter.classList.toggle('meter--indeterminate', indeterminate);
    if (indeterminate) {
      el.meter.removeAttribute('aria-valuenow');
      el.meterFill.style.width = '';
      return;
    }
    frac = Math.max(0, Math.min(1, frac));
    el.meterFill.style.width = (frac * 100).toFixed(1) + '%';
    el.meter.setAttribute('aria-valuenow', String(Math.round(frac * 100)));
  }

  /**
   * Sets the badge text and its state modifier class.
   * @param {string} phase
   * @param {string} label
   * @param {string} text
   */
  function setBadge(phase, label, text) {
    el.badge.textContent = label;
    el.badge.className = 'live-status__badge live-status__badge--' + phase;
    el.text.textContent = text;
  }

  /**
   * Renders the upload phase (before the server has a job id).
   * @param {number} frac  0..1
   */
  function renderUpload(frac) {
    var f = app.format;
    paintLadder('upload', false, false);
    setBadge('uploading', 'Uploading', 'sending the recording to the server — ' + f.percent(frac));
    el.elapsed.textContent = f.clock((Date.now() - timing.submittedAt) / 1000);
    el.done.textContent = '—';
    el.speed.textContent = '—';
    el.eta.textContent = '—';
    setMeter(frac);
  }

  /**
   * Renders a polled job status from GET /api/jobs/{id}.
   * @param {Object} job
   */
  function renderJob(job) {
    var f = app.format;
    var now = Date.now();
    var status = job.status;
    var stage = job.stage;
    var done = job.seconds_done || 0;
    var dur = job.duration;
    var runSec, speed, frac;

    el.elapsed.textContent = f.clock((now - timing.submittedAt) / 1000);

    if (status === 'queued') {
      paintLadder('queued', false, false);
      setBadge('queued', 'Queued', stageText('queued', job));
      setMeter(null);
      return;
    }
    if (status === 'running') {
      if (!timing.runStartedAt) {
        timing.runStartedAt = now;
      }
      paintLadder(stage, false, false);
      setBadge('running', 'Running', stageText(stage, job));
      runSec = (now - timing.runStartedAt) / 1000;
      speed = runSec > 2 && done > 0 ? done / runSec : null;
      el.done.textContent = f.clock(done) + (dur ? ' / ' + f.clock(dur) : '');
      el.speed.textContent = speed ? speed.toFixed(1) + '× real time' : '—';
      if (stage === 'frames') {
        frac = dur ? done / dur : null;
        setMeter(frac === null ? null : Math.min(frac, 0.99));
        el.eta.textContent = dur && speed ? f.clock((dur - done) / speed) : '—';
      } else {
        setMeter(null);
        el.eta.textContent = 'finishing';
      }
      return;
    }
    if (status === 'done') {
      paintLadder('report', true, false);
      setBadge('done', 'Finished', 'analysis complete');
      setMeter(1);
      el.eta.textContent = '0:00';
      if (job.result && job.result.stats) {
        // A job reopened from its #job= link has no client-side clock; fall back
        // to the server's processing time so Elapsed never reads 0:00.
        el.elapsed.textContent = f.clock(Math.max((now - timing.submittedAt) / 1000,
          job.result.stats.processing_seconds));
        el.done.textContent = f.clock(job.result.stats.video_seconds);
        el.speed.textContent = f.fixed(job.result.stats.speed_x_realtime, 1) + '× real time';
      }
      return;
    }
    if (status === 'cancelled') {
      paintLadder(stage, false, true);
      setBadge('cancelled', 'Cancelled', 'job cancelled — the upload was deleted');
      setMeter(0);
      return;
    }
    // error
    paintLadder(STEPS.indexOf(stage) >= 0 ? stage : 'frames', false, true);
    setBadge('error', 'Error', 'see the message below');
    setMeter(0);
  }

  /**
   * Starts the client-side clock for a new submission.
   * @param {string} device torch device the server reports for the CNN
   */
  function start(device) {
    cnnDevice = device || 'cpu';
    timing.submittedAt = Date.now();
    timing.runStartedAt = 0;
  }

  /** Caches element references. */
  function init() {
    el.ladder = document.getElementById('ladder');
    el.badge = document.getElementById('live-badge');
    el.text = document.getElementById('live-text');
    el.elapsed = document.getElementById('live-elapsed');
    el.done = document.getElementById('live-done');
    el.speed = document.getElementById('live-speed');
    el.eta = document.getElementById('live-eta');
    el.meter = document.getElementById('meter');
    el.meterFill = document.getElementById('meter-fill');
  }

  app.progress = { init: init, start: start, renderUpload: renderUpload, renderJob: renderJob };
})(window.GazeApp = window.GazeApp || {});
