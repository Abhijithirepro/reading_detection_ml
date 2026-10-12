/* format.js — pure formatting helpers shared by every component. */
(function (app) {
  'use strict';

  /**
   * Formats seconds as m:ss (elapsed / remaining clocks).
   * @param {number} sec
   * @returns {string}
   */
  function clock(sec) {
    var s, m, r;
    s = Math.max(0, Math.floor(sec || 0));
    m = Math.floor(s / 60);
    r = s % 60;
    return m + ':' + (r < 10 ? '0' : '') + r;
  }

  /**
   * Formats seconds as mm:ss — matches predict.py's mmss() so window ranges read
   * the same here as in the CLI output and the JSON.
   * @param {number} sec
   * @returns {string}
   */
  function mmss(sec) {
    var total, m, s;
    total = Math.round(sec || 0);
    m = Math.floor(total / 60);
    s = total % 60;
    return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
  }

  /**
   * Fixed-point number, or an em dash when the value is missing.
   * @param {number|null} n
   * @param {number} digits
   * @returns {string}
   */
  function fixed(n, digits) {
    if (n === null || n === undefined || isNaN(n)) {
      return '—';
    }
    return Number(n).toFixed(digits);
  }

  /**
   * Fraction as a whole percentage, e.g. 0.874 -> "87%".
   * @param {number} frac
   * @returns {string}
   */
  function percent(frac) {
    if (frac === null || frac === undefined || isNaN(frac)) {
      return '—';
    }
    return Math.round(frac * 100) + '%';
  }

  app.format = { clock: clock, mmss: mmss, fixed: fixed, percent: percent };
})(window.GazeApp = window.GazeApp || {});
