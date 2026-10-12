/* chart.js — per-window probability over time, as an inline SVG bar chart.
   One measure on one axis. Colour carries a state (below vs above the flag
   threshold) using the reserved status red; the legend and the window table keep
   that distinction off colour alone. Same drawing as frontend/js/app.js. */
(function (app) {
  'use strict';

  var W = 1000;
  var H = 210;
  var PAD = { t: 10, r: 8, b: 26, l: 38 };

  /**
   * Y pixel for a probability.
   * @param {number} p
   * @returns {number}
   */
  function yOf(p) {
    return PAD.t + (H - PAD.t - PAD.b) * (1 - p);
  }

  /**
   * Builds the SVG markup. Every interpolated value is a number or a server-side
   * time string, so string-built markup is safe here.
   * @param {Array<Object>} windows  [{start_s, end_s, prob}]
   * @param {number} threshold
   * @returns {string}
   */
  function buildSvg(windows, threshold) {
    var mmss = app.format.mmss;
    var plotW = W - PAD.l - PAD.r;
    var plotH = H - PAD.t - PAD.b;
    var n = windows.length;
    var slot = plotW / n;
    var barW = Math.max(1, slot - (slot > 5 ? 2 : 0.5));
    var parts = [];
    var grid = [0, 0.25, 0.5, 0.75, 1];
    var i, w, p, h, x, hit, range;

    parts.push('<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Suspicion score for each of ' + n + ' windows.">');

    for (i = 0; i < grid.length; i++) {
      parts.push(
        '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + yOf(grid[i]) + '" y2="' + yOf(grid[i]) +
        '" stroke="var(--grid)" stroke-width="1"/>' +
        '<text x="' + (PAD.l - 8) + '" y="' + (yOf(grid[i]) + 3.5) + '" text-anchor="end" font-size="10" ' +
        'fill="var(--ink-mute)" font-family="var(--mono)">' + grid[i].toFixed(2) + '</text>'
      );
    }

    for (i = 0; i < n; i++) {
      w = windows[i];
      p = w.prob;
      hit = p >= threshold;
      h = Math.max(1.5, plotH * p);
      x = PAD.l + i * slot + (slot - barW) / 2;
      range = mmss(w.start_s) + '–' + mmss(w.end_s);
      parts.push(
        '<rect class="bar" x="' + x.toFixed(2) + '" y="' + (PAD.t + plotH - h).toFixed(2) + '" ' +
        'width="' + barW.toFixed(2) + '" height="' + h.toFixed(2) + '" rx="' + (barW > 5 ? 2 : 0) + '" ' +
        'fill="' + (hit ? 'var(--series-flag)' : 'var(--series-base)') + '" ' +
        'data-i="' + i + '" data-p="' + p.toFixed(3) + '" data-r="' + range + '">' +
        '<title>' + range + ': ' + p.toFixed(3) + '</title></rect>'
      );
    }

    parts.push(
      '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + yOf(threshold) + '" y2="' + yOf(threshold) +
      '" stroke="var(--amber)" stroke-width="2" stroke-dasharray="6 4"/>' +
      '<line x1="' + PAD.l + '" x2="' + (W - PAD.r) + '" y1="' + yOf(0) + '" y2="' + yOf(0) +
      '" stroke="var(--axis)" stroke-width="1"/>' +
      '<text x="' + PAD.l + '" y="' + (H - 8) + '" font-size="10" fill="var(--ink-mute)" font-family="var(--mono)">00:00</text>' +
      '<text x="' + (W - PAD.r) + '" y="' + (H - 8) + '" text-anchor="end" font-size="10" fill="var(--ink-mute)" ' +
      'font-family="var(--mono)">' + mmss(windows[n - 1].end_s) + '</text>'
    );
    parts.push('</svg>');
    return parts.join('');
  }

  /**
   * Draws the chart and wires the hover tooltip and bar clicks.
   * @param {HTMLElement} host
   * @param {HTMLElement} tip
   * @param {Array<Object>} windows
   * @param {number} threshold
   * @param {function(number)=} onSelect  called with a bar's window index on click
   */
  function draw(host, tip, windows, threshold, onSelect) {
    var svg;
    if (!host || !windows.length) {
      return;
    }
    host.innerHTML = buildSvg(windows, threshold);
    svg = host.querySelector('svg');

    if (onSelect) {
      svg.addEventListener('click', function (e) {
        var bar = e.target.closest ? e.target.closest('rect.bar') : null;
        if (bar) {
          onSelect(parseInt(bar.getAttribute('data-i'), 10));
        }
      });
    }

    svg.addEventListener('mousemove', function (e) {
      var bar = e.target.closest ? e.target.closest('rect.bar') : null;
      var box, strong, span;
      if (!bar) {
        tip.hidden = true;
        return;
      }
      box = host.getBoundingClientRect();
      tip.textContent = '';
      strong = document.createElement('b');
      strong.textContent = bar.getAttribute('data-p');
      span = document.createElement('span');
      span.textContent = bar.getAttribute('data-r');
      tip.appendChild(strong);
      tip.appendChild(span);
      tip.hidden = false;
      tip.style.left = (e.clientX - box.left) + 'px';
      tip.style.top = (e.clientY - box.top) + 'px';
    });
    svg.addEventListener('mouseleave', function () { tip.hidden = true; });
  }

  app.chart = { draw: draw };
})(window.GazeApp = window.GazeApp || {});
