/* http.js — XMLHttpRequest wrappers. Node-style callbacks: cb(err, data). */
(function (app) {
  'use strict';

  /**
   * Pulls the server's error message out of a FastAPI error body ({detail: ...}).
   * @param {XMLHttpRequest} xhr
   * @returns {Error}
   */
  function toError(xhr) {
    var msg, body;
    msg = 'Request failed (HTTP ' + xhr.status + ')';
    try {
      body = JSON.parse(xhr.responseText);
      if (body && typeof body.detail === 'string') {
        msg = body.detail;
      }
    } catch (e) {
      // non-JSON body: keep the generic message
    }
    if (xhr.status === 0) {
      msg = 'Could not reach the web server hosting this page.';
    }
    return new Error(msg);
  }

  /**
   * Sends a request and parses a JSON response.
   * @param {string} method
   * @param {string} url
   * @param {FormData|null} body
   * @param {function(Error, Object)} cb
   * @param {function(number)=} onUpload  upload fraction 0..1 (POST only)
   * @returns {XMLHttpRequest} so the caller can abort() it
   */
  function request(method, url, body, cb, onUpload) {
    var xhr = new XMLHttpRequest();
    xhr.open(method, url, true);
    if (onUpload && xhr.upload) {
      xhr.upload.addEventListener('progress', function (e) {
        if (e.lengthComputable) {
          onUpload(e.loaded / e.total);
        }
      });
    }
    xhr.onreadystatechange = function () {
      var data;
      if (xhr.readyState !== 4) {
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          data = JSON.parse(xhr.responseText);
        } catch (e) {
          cb(new Error('Malformed server response'), null);
          return;
        }
        cb(null, data);
      } else if (xhr.status !== 0 || !xhr.aborted) {
        cb(toError(xhr), null);
      }
    };
    xhr.send(body);
    return xhr;
  }

  /**
   * GET a JSON resource.
   * @param {string} url
   * @param {function(Error, Object)} cb
   * @returns {XMLHttpRequest}
   */
  function getJson(url, cb) {
    return request('GET', url, null, cb);
  }

  /**
   * POST multipart form data, reporting upload progress.
   * @param {string} url
   * @param {FormData} form
   * @param {function(number)} onUpload
   * @param {function(Error, Object)} cb
   * @returns {XMLHttpRequest}
   */
  function postForm(url, form, onUpload, cb) {
    return request('POST', url, form, cb, onUpload);
  }

  /**
   * DELETE a resource.
   * @param {string} url
   * @param {function(Error, Object)} cb
   * @returns {XMLHttpRequest}
   */
  function del(url, cb) {
    return request('DELETE', url, null, cb);
  }

  /**
   * Abort an in-flight request without reporting it as a failure.
   * @param {XMLHttpRequest} xhr
   */
  function abort(xhr) {
    if (xhr) {
      xhr.aborted = true;
      xhr.abort();
    }
  }

  app.http = { getJson: getJson, postForm: postForm, del: del, abort: abort };
})(window.GazeApp = window.GazeApp || {});
