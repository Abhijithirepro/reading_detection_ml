/* library.js (v2) — the pre-scored candidates: every recording in the three
   folders, scored ahead of time by webapp/library.py. A folder filter, search, a
   "disagrees with folder" filter and sortable columns; a row opens the candidate
   (main.js routes #cand=<id>). Prev / Next walk the list as currently filtered.
   User-derived text goes in via textContent only. */
(function (app) {
  'use strict';

  var FOLDER_TEXT = {
    cheating_folder: 'cheating',
    not_cheating_folder: 'not cheating',
    new_not_cheating: 'new not cheating'
  };

  var el = {};
  var state = {
    data: null, rows: [], folder: 'all', query: '', mismatch: false,
    sort: null, dir: 1, onOpen: null, lastId: null
  };

  /**
   * @param {string} tag
   * @param {string} cls
   * @param {string=} text
   * @returns {HTMLElement}
   */
  function make(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) {
      node.className = cls;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  /**
   * Rows passing the folder / search / mismatch filters, in the chosen order.
   * @returns {Array<Object>}
   */
  function visibleRows() {
    var q = state.query.toLowerCase();
    var out = state.rows.filter(function (r) {
      if (state.folder !== 'all' && r.folder !== state.folder) {
        return false;
      }
      if (state.mismatch && r.correct) {
        return false;
      }
      return !q || r.name.toLowerCase().indexOf(q) >= 0 || ('ir' + r.ir).indexOf(q) >= 0;
    });
    if (state.sort) {
      out.sort(function (a, b) {
        var x = a[state.sort];
        var y = b[state.sort];
        if (typeof x === 'string') {
          return state.dir * x.localeCompare(y);
        }
        return state.dir * ((x === y) ? 0 : (x < y ? -1 : 1));
      });
    }
    return out;
  }

  /** Folder buttons with counts. */
  function renderTabs() {
    var tabs = [{ key: 'all', text: 'All', n: state.rows.length }];
    var i, f, btn;
    for (i = 0; i < state.data.folders.length; i++) {
      f = state.data.folders[i];
      tabs.push({ key: f.folder, text: f.folder, n: f.n });
    }
    el.tabs.textContent = '';
    for (i = 0; i < tabs.length; i++) {
      btn = make('button', 'chip' + (tabs[i].key === state.folder ? ' is-on' : ''));
      btn.type = 'button';
      btn.setAttribute('data-folder', tabs[i].key);
      btn.setAttribute('aria-pressed', String(tabs[i].key === state.folder));
      btn.appendChild(document.createTextNode(tabs[i].text + ' '));
      btn.appendChild(make('span', 'chip__n', String(tabs[i].n)));
      el.tabs.appendChild(btn);
    }
  }

  /** The table body for the current filters. */
  function renderRows() {
    var f = app.format;
    var rows = visibleRows();
    var thr = state.data.video_threshold;
    var frag = document.createDocumentFragment();
    var i, r, tr, td, name, bar, flagged;
    for (i = 0; i < rows.length; i++) {
      r = rows[i];
      flagged = r.verdict === 'CHEATING';
      tr = make('tr', 'lib-row' + (r.correct ? '' : ' is-mismatch'));
      tr.tabIndex = 0;
      tr.setAttribute('data-id', r.id);

      td = make('td', 'lib-cand');
      name = make('span', 'lib-cand__name', r.name);
      td.appendChild(name);
      td.appendChild(make('span', 'lib-cand__ir', 'ir' + r.ir + (r.reviewed ? ' · reviewed' : '')));
      tr.appendChild(td);

      td = make('td');
      td.appendChild(make('span', 'tag tag--' + (r.actual === 'cheating' ? 'cheat' : 'clean'),
        FOLDER_TEXT[r.folder] || r.folder));
      tr.appendChild(td);

      td = make('td');
      td.appendChild(make('span', 'pill pill--' + (flagged ? 'flag' : 'clear'), flagged ? 'Reading' : 'Clear'));
      tr.appendChild(td);

      td = make('td', 'num lib-score');
      bar = make('span', 'lib-score__bar' + (flagged ? ' is-flagged' : ''));
      bar.style.width = (Math.min(1, r.score / (2 * thr)) * 100) + '%';
      td.appendChild(make('span', 'lib-score__track')).appendChild(bar);
      td.appendChild(make('span', 'lib-score__val', f.fixed(r.score, 3)));
      tr.appendChild(td);

      tr.appendChild(make('td', 'num', f.clock(r.reading_seconds)));
      tr.appendChild(make('td', 'num', String(r.segments)));
      tr.appendChild(make('td', 'num', f.clock(r.media_seconds || r.duration)));
      td = make('td', 'lib-agree ' + (r.correct ? 'is-yes' : 'is-no'), r.correct ? '✓' : '✗');
      td.setAttribute('aria-label', r.correct ? 'agrees with folder' : 'disagrees with folder');
      tr.appendChild(td);
      frag.appendChild(tr);
    }
    el.rows.textContent = '';
    el.rows.appendChild(frag);
    el.empty.hidden = rows.length > 0;
    el.empty.textContent = 'No candidate matches these filters.';
    markSort();
  }

  /** Arrow on the sorted column header. */
  function markSort() {
    var btns = el.table.querySelectorAll('th button[data-sort]');
    var i, key;
    for (i = 0; i < btns.length; i++) {
      key = btns[i].getAttribute('data-sort');
      btns[i].parentNode.setAttribute('aria-sort', key === state.sort
        ? (state.dir > 0 ? 'ascending' : 'descending') : 'none');
      btns[i].classList.toggle('is-sorted', key === state.sort);
      btns[i].setAttribute('data-dir', key === state.sort ? (state.dir > 0 ? '↑' : '↓') : '');
    }
  }

  /** Headline numbers + the scoring note. */
  function renderSummary() {
    var d = state.data;
    var s = d.summary;
    el.summary.textContent = s.n + ' recordings · cheating caught ' + s.caught + ' / ' + s.cheating +
      ' · clean cleared ' + s.cleared + ' / ' + s.clean;
    el.note.textContent = 'Scored out-of-fold: every recording by a model that never trained on it ' +
      '(the cross-validation behind ' + d.experiment + '), so these are the honest numbers. ' +
      'Model says “Reading” when the average reading probability while the candidate speaks is ≥ ' +
      app.format.fixed(d.video_threshold, 3) + ' (the score column). Folder = how the recording was labelled.';
  }

  /**
   * Opens the candidate a row stands for.
   * @param {Event} e
   */
  function onRowActivate(e) {
    var row = e.target.closest ? e.target.closest('tr[data-id]') : null;
    if (!row || (e.type === 'keydown' && e.key !== 'Enter' && e.key !== ' ')) {
      return;
    }
    e.preventDefault();
    if (state.onOpen) {
      state.onOpen(row.getAttribute('data-id'));
    }
  }

  /**
   * Loads GET /api/library and renders the panel.
   * @param {function(Error=)=} done
   */
  function load(done) {
    app.http.getJson('/api/library', function (err, data) {
      if (err) {
        el.summary.textContent = 'Library unavailable: ' + err.message;
        if (done) { done(err); }
        return;
      }
      state.data = data;
      state.rows = data.candidates || [];
      if (!state.rows.length) {
        el.summary.textContent = data.hint || 'The library is empty.';
        el.tools.hidden = true;
        el.scroll.hidden = true;
        if (done) { done(); }
        return;
      }
      renderSummary();
      renderTabs();
      renderRows();
      if (done) { done(); }
    });
  }

  /**
   * The library row for a candidate id, or null.
   * @param {string} id
   * @returns {Object|null}
   */
  function find(id) {
    var i;
    for (i = 0; i < state.rows.length; i++) {
      if (state.rows[i].id === id) {
        return state.rows[i];
      }
    }
    return null;
  }

  /**
   * Position of a candidate in the list as filtered now, and its neighbours. Falls
   * back to the full list when the candidate is filtered out.
   * @param {string} id
   * @returns {{pos: number, total: number, prev: (string|null), next: (string|null)}}
   */
  function neighbours(id) {
    var list = visibleRows();
    var i = -1;
    var k;
    for (k = 0; k < list.length; k++) {
      if (list[k].id === id) { i = k; break; }
    }
    if (i < 0) {
      list = state.rows;
      for (k = 0; k < list.length; k++) {
        if (list[k].id === id) { i = k; break; }
      }
    }
    return {
      pos: i + 1, total: list.length,
      prev: i > 0 ? list[i - 1].id : null,
      next: i >= 0 && i < list.length - 1 ? list[i + 1].id : null
    };
  }

  /**
   * Scrolls the list to the candidate last opened and focuses its row.
   * @param {string} id
   */
  function reveal(id) {
    var row = id ? el.rows.querySelector('tr[data-id="' + id + '"]') : null;
    if (row) {
      row.scrollIntoView({ block: 'center' });
      row.focus({ preventScroll: true });
    }
  }

  /**
   * @param {{onOpen: function(string)}} opts
   */
  function init(opts) {
    var ids = {
      panel: 'panel-library', summary: 'lib-summary', tabs: 'lib-tabs', search: 'lib-search',
      mismatchBox: 'lib-mismatch', table: 'lib-table', rows: 'lib-rows', empty: 'lib-empty',
      note: 'lib-note'
    };
    var key;
    for (key in ids) {
      if (Object.prototype.hasOwnProperty.call(ids, key)) {
        el[key] = document.getElementById(ids[key]);
      }
    }
    el.scroll = el.table.parentNode;
    el.tools = el.tabs.parentNode;
    state.onOpen = opts.onOpen;

    el.tabs.addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('[data-folder]') : null;
      if (btn) {
        state.folder = btn.getAttribute('data-folder');
        renderTabs();
        renderRows();
      }
    });
    el.search.addEventListener('input', function () {
      state.query = el.search.value.trim();
      renderRows();
    });
    el.mismatchBox.addEventListener('change', function () {
      state.mismatch = el.mismatchBox.checked;
      renderRows();
    });
    el.table.querySelector('thead').addEventListener('click', function (e) {
      var btn = e.target.closest ? e.target.closest('button[data-sort]') : null;
      var key2;
      if (!btn) {
        return;
      }
      key2 = btn.getAttribute('data-sort');
      if (state.sort === key2) {
        state.dir = -state.dir;
      } else {
        state.sort = key2;
        state.dir = (key2 === 'name' || key2 === 'folder') ? 1 : -1;
      }
      renderRows();
    });
    el.rows.addEventListener('click', onRowActivate);
    el.rows.addEventListener('keydown', onRowActivate);
  }

  app.library = { init: init, load: load, find: find, neighbours: neighbours, reveal: reveal };
})(window.GazeApp = window.GazeApp || {});
