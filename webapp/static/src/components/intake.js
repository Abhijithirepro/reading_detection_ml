/* intake.js — the "Specimen" panel: drop zone, file picker, minutes cap, Analyse. */
(function (app) {
  'use strict';

  var el = {};
  var state = { file: null, ready: false, accepted: [] };
  var handlers = { onSubmit: null };

  /**
   * Formats a byte count for the filename line.
   * @param {number} bytes
   * @returns {string}
   */
  function humanSize(bytes) {
    if (bytes >= 1e9) {
      return (bytes / 1e9).toFixed(2) + ' GB';
    }
    return (bytes / 1e6).toFixed(1) + ' MB';
  }

  /**
   * True when the file's extension is one the server accepts.
   * @param {File} file
   * @returns {boolean}
   */
  function isAccepted(file) {
    var dot = file.name.lastIndexOf('.');
    var ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
    return !state.accepted.length || state.accepted.indexOf(ext) !== -1;
  }

  /**
   * Shows or clears the intake error line.
   * @param {string|null} msg
   */
  function showError(msg) {
    el.error.textContent = msg || '';
    el.error.hidden = !msg;
  }

  /** Enables Analyse only when the server is up and a file is chosen. */
  function refresh() {
    el.run.disabled = !(state.ready && state.file);
  }

  /**
   * Takes a dropped or picked file.
   * @param {File|undefined} file
   */
  function acceptFile(file) {
    if (!file) {
      return;
    }
    if (!isAccepted(file)) {
      showError('Unsupported file type. Use one of: ' + state.accepted.join(', '));
      return;
    }
    showError(null);
    state.file = file;
    el.filename.textContent = file.name + ' · ' + humanSize(file.size);
    el.drop.classList.add('is-loaded');
    refresh();
  }

  /**
   * Minutes cap from the input, or null for the whole recording.
   * @returns {number|null}
   */
  function maxMinutes() {
    var v = parseFloat(el.maxmin.value);
    return isFinite(v) && v > 0 ? v : null;
  }

  /** Wires up the drop zone, picker and button. */
  function bind() {
    el.drop.addEventListener('click', function () { el.file.click(); });
    el.drop.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        el.file.click();
      }
    });
    el.file.addEventListener('change', function () { acceptFile(el.file.files[0]); });

    ['dragenter', 'dragover'].forEach(function (type) {
      el.drop.addEventListener(type, function (e) {
        e.preventDefault();
        el.drop.classList.add('is-over');
      });
    });
    ['dragleave', 'drop'].forEach(function (type) {
      el.drop.addEventListener(type, function (e) {
        e.preventDefault();
        el.drop.classList.remove('is-over');
      });
    });
    el.drop.addEventListener('drop', function (e) {
      var files = e.dataTransfer && e.dataTransfer.files;
      acceptFile(files && files[0]);
    });

    el.run.addEventListener('click', function () {
      if (state.file && handlers.onSubmit) {
        handlers.onSubmit(state.file, maxMinutes());
      }
    });
  }

  /**
   * Initialises the panel.
   * @param {{onSubmit: function(File, number|null)}} opts
   */
  function init(opts) {
    el.drop = document.getElementById('drop');
    el.file = document.getElementById('file');
    el.run = document.getElementById('run');
    el.maxmin = document.getElementById('maxmin');
    el.filename = document.getElementById('filename');
    el.error = document.getElementById('intake-error');
    el.accepted = document.getElementById('accepted');
    handlers.onSubmit = opts.onSubmit;
    if (el.drop && el.file && el.run) {
      bind();
    }
  }

  /**
   * Marks the server as reachable and records the accepted extensions.
   * @param {Array<string>} accepted e.g. ['.mp4', '.webm']
   */
  function setReady(accepted) {
    state.ready = true;
    state.accepted = accepted || [];
    if (el.accepted && state.accepted.length) {
      el.accepted.textContent = state.accepted.map(function (x) { return x.slice(1); }).join(', ');
    }
    refresh();
  }

  /** Clears the chosen file for a fresh run. */
  function reset() {
    state.file = null;
    el.file.value = '';
    el.filename.textContent = '';
    el.drop.classList.remove('is-loaded');
    showError(null);
    refresh();
  }

  app.intake = { init: init, setReady: setReady, reset: reset, showError: showError };
})(window.GazeApp = window.GazeApp || {});
