/* compare.js (v2) — agreement between the detected reading segments and a manual
   review. The server attaches a review automatically when the recording's IR
   number matches one in manual_review_for_candidate/; any review CSV in the same
   sheet format can also be loaded here. Reviewer spans are jump buttons, marked
   caught / missed by how much of each span the detector flagged. User-derived
   text goes in via textContent only. */
(function (app) {
  'use strict';

  var el = {};
  var state = { jobId: null };

  /**
   * Renders a comparison (payload.review) or the empty state.
   * @param {Object|null} review
   */
  function show(review) {
    var f = app.format;
    var m, i, s, li, btn, range, cov, note, extra;
    el.spans.textContent = '';
    el.extra.textContent = '';
    el.extraH.hidden = true;
    el.error.hidden = true;
    if (!review) {
      el.src.textContent = 'No manual review for this recording yet.';
      el.metrics.hidden = true;
      return;
    }
    m = review.metrics;
    el.src.textContent = review.source + ' · ' + (review.origin || 'uploaded');
    el.metrics.hidden = false;
    el.spansCaught.textContent = m.spans_caught;
    el.auc.textContent = f.fixed(m.auc, 3);
    el.p.textContent = f.percent(m.precision);
    el.r.textContent = f.percent(m.recall);
    for (i = 0; i < review.spans.length; i++) {
      s = review.spans[i];
      li = document.createElement('li');
      btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'cmp-span ' + (s.caught ? 'is-caught' : 'is-missed');
      btn.setAttribute('data-t', String(s.start_s));
      range = document.createElement('span');
      range.className = 'cmp-span__range';
      range.textContent = (s.caught ? '✓ ' : '✗ ') + f.mmss(s.start_s) + '–' + f.mmss(s.end_s);
      cov = document.createElement('span');
      cov.className = 'cmp-span__cov';
      cov.textContent = f.percent(s.coverage) + ' flagged';
      note = document.createElement('span');
      note.className = 'cmp-span__note';
      note.textContent = s.reason;
      btn.appendChild(range);
      btn.appendChild(cov);
      btn.appendChild(note);
      li.appendChild(btn);
      el.spans.appendChild(li);
    }

    // detections the reviewer did not mark: false alarms, or reading worth a second look
    extra = review.extra_segments || [];
    el.extraH.hidden = extra.length === 0;
    el.extraH.textContent = 'Flagged by the model, not in the review (' + extra.length + ') — worth a look:';
    for (i = 0; i < extra.length; i++) {
      s = extra[i];
      li = document.createElement('li');
      btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'cmp-span is-extra';
      btn.setAttribute('data-t', String(s.start_s));
      range = document.createElement('span');
      range.className = 'cmp-span__range';
      range.textContent = '? ' + f.mmss(s.start_s) + '–' + f.mmss(s.end_s);
      btn.appendChild(range);
      li.appendChild(btn);
      el.extra.appendChild(li);
    }
  }

  /**
   * Called by result.js when a job is rendered.
   * @param {Object} payload
   */
  function render(payload) {
    state.jobId = payload.job_id;
    show(payload.review || null);
  }

  /** Uploads the chosen review CSV and re-renders with the server's comparison. */
  function upload(file) {
    var form = new FormData();
    if (!state.jobId) {
      return;
    }
    form.append('file', file, file.name);
    el.pick.disabled = true;
    app.http.postForm('/api/jobs/' + state.jobId + '/review', form, function () {}, function (err, review) {
      el.pick.disabled = false;
      el.file.value = '';
      if (err) {
        el.error.textContent = err.message;
        el.error.hidden = false;
        return;
      }
      show(review);
      app.review.setReview(review);
    });
  }

  /** Caches element references and binds events. */
  function init() {
    var ids = {
      src: 'compare-src', metrics: 'compare-metrics', spans: 'compare-spans', pick: 'compare-pick',
      file: 'compare-file', error: 'compare-error', spansCaught: 'cmp-spans', auc: 'cmp-auc',
      p: 'cmp-p', r: 'cmp-r', extra: 'compare-extra', extraH: 'compare-extra-h'
    };
    var key;
    for (key in ids) {
      if (Object.prototype.hasOwnProperty.call(ids, key)) {
        el[key] = document.getElementById(ids[key]);
      }
    }
    if (!el.pick) {
      return;
    }
    el.pick.addEventListener('click', function () { el.file.click(); });
    el.file.addEventListener('change', function () {
      if (el.file.files[0]) {
        upload(el.file.files[0]);
      }
    });
    function jump(e) {
      var btn = e.target.closest ? e.target.closest('.cmp-span') : null;
      if (btn) {
        app.review.seekTo(parseFloat(btn.getAttribute('data-t')), true, true);
      }
    }
    el.spans.addEventListener('click', jump);
    el.extra.addEventListener('click', jump);
  }

  app.compare = { init: init, render: render };
})(window.GazeApp = window.GazeApp || {});
