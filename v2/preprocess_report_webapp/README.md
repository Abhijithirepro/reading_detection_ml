# preprocess_report_webapp — the pre-scored candidate page, ready to host

A self-contained copy of the v2 web app's **candidate library** (the page on
`http://127.0.0.1:8002`): all 100 candidates with their verdict, video, reading timeline,
reasons, eye / head / mouth traces, manual-review comparison and detailed report.

Everything is plain files, so **any web server can host it**: nothing is analysed on
the server, no Python model, no MediaPipe, no `pip install`.

| Path | What it is | Size |
|---|---|---|
| `index.html`, `src/` | the page (same as `v2/webapp/static`, reading files instead of `/api/…`) | 170 KB |
| `data/info.json` | model facts for the header | |
| `data/index.json` | the candidate list | |
| `data/<id>/payload.json` | one candidate's result (verdict, per-second scores, segments, reasons) | |
| `data/<id>/traces.json` | eye / head / mouth traces for the live chart | |
| `data/<id>/report.html` | the detailed report | |
| `data/<id>/segments.csv` | reading segments in the reviewers' sheet format | 146 MB in total |
| `videos/<id>.mp4` / `.webm` | the 100 recordings (34 mp4 + 66 seekable webm) | **≈ 15 GB** |
| `serve.py` | optional server, Python standard library only | |
| `build_static.py` | rebuilds `data/` and `videos/` on the Mac — **not needed on the server** | |

Not included: **analysing a new recording** (the upload box). That needs the full
pipeline — run `v2/webapp/server.py` for it.

> **Privacy:** this folder holds candidates' names and interview recordings, and the page
> has **no login**. Keep it on an internal network / VPN, or put a password in front of it
> (nginx example below).

---

## 1. Copy the folder to the server

Copy the **whole folder**, including `data/` and `videos/` (≈ 15 GB):

```bash
rsync -av --progress "v2/preprocess_report_webapp/" user@your-server:/var/www/preprocess_report_webapp/
```

(`scp -r` or any file copy works too; rsync only re-sends what changed next time.)

## 2a. Host it with nginx (recommended)

nginx serves video byte ranges out of the box, so seeking in the videos works.

```nginx
server {
    listen 80;
    server_name your-server;

    root  /var/www/preprocess_report_webapp;
    index index.html;
    autoindex off;                                   # never list folders

    location ~ ^/(serve\.py|build_static\.py|README\.md|\.gitignore)$ { deny all; }

    # optional password:  sudo htpasswd -c /etc/nginx/.htpasswd someone
    # auth_basic           "Reading detector";
    # auth_basic_user_file /etc/nginx/.htpasswd;
}
```

All links in the page are relative, so it also works under a sub-path, e.g.
`location /preprocess_report_webapp/ { alias /var/www/preprocess_report_webapp/; }` →
`https://your-server/preprocess_report_webapp/`.

Apache works the same way (`DocumentRoot` pointing at the folder, `Options -Indexes`).

## 2b. …or with the included Python server

No nginx? Python 3.8+ is enough — nothing to install:

```bash
cd /var/www/preprocess_report_webapp
python3 serve.py --host 0.0.0.0 --port 8002        # http://your-server:8002
```

- Without `--host 0.0.0.0` it only answers on the server itself (127.0.0.1).
- It supports video seeking (HTTP Range) and refuses folder listings, unlike
  `python3 -m http.server`.
- To keep it running after you log out: `nohup python3 serve.py --host 0.0.0.0 --quiet &`,
  or a systemd service with `ExecStart=/usr/bin/python3 /var/www/preprocess_report_webapp/serve.py --host 0.0.0.0 --quiet`.

## 3. Open it

`http://your-server/` (nginx) or `http://your-server:8002/` (serve.py). A candidate can be
linked directly: `http://your-server/#cand=ir8266049`.

---

## Updating after the library is rebuilt

On the Mac, after `venv/bin/python v2/webapp/library.py` has rebuilt the library:

```bash
python3 v2/preprocess_report_webapp/build_static.py              # data + any new / changed videos
python3 v2/preprocess_report_webapp/build_static.py --no-videos  # data only
```

then copy the folder again (rsync sends only the changed files). `build_static.py` only
reads `v2/webapp/library/` and `v2/models/model.json`, and writes only inside this folder.
On macOS the videos are APFS clones of the library's files, so they take no extra disk
space on the Mac until one of them is changed.

`data/` and `videos/` are git-ignored (see `.gitignore`): they are generated and large.
