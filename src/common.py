"""
common.py — shared config, signal processing, and gaze-extraction backends.

This module is the backbone of the gaze cheat detector. It defines:

  * Config            — every tunable constant (window size, thresholds, landmark
                        indices, sampling rate). JSON-serializable so a future
                        in-browser JS port reads the same numbers.
  * signal utilities  — cleaning, gap interpolation, median filtering, resampling.
  * saccade/sweep      — velocity-based saccade segmentation and the reading-sweep
                        detector that is the heart of the "reading an overlay" signal.
  * gaze backends     — one interface, three implementations:
                          MediaPipeBackend  (always on: iris gaze, head pose, blinks)
                          L2CSBackend       (appearance gaze yaw/pitch; default extra)
                          OpenFace3Backend  (alternative appearance backend)

Heavy deps (mediapipe, torch, cv2) are imported lazily so that pure-signal code
(and `--selftest`) runs without a camera stack installed.

Per-frame time-series schema (one parquet row per sampled frame):
    t, frame_idx, face_found,
    gaze_x, gaze_y,          # iris-in-eye, normalized to [-1, 1] (MediaPipe)
    yaw, pitch, roll,        # head pose, degrees
    ear, blink,              # eye aspect ratio + blink flag
    [gaze_yaw_l2cs, gaze_pitch_l2cs]   # optional appearance channels
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
WINDOWS_DIR = DATA_DIR / "windows"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Per-frame columns that hold gaze/pose signal (used across stages).
BASE_CHANNELS = ["gaze_x", "gaze_y", "yaw", "pitch", "roll", "ear", "mouth"]
L2CS_CHANNELS = ["gaze_yaw_l2cs", "gaze_pitch_l2cs"]

# CNN input channels. Kept small + deployable.
SEQ_CHANNELS = ["gaze_x", "gaze_y", "yaw", "pitch", "ear", "mouth"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """All tunable constants. Saved as JSON next to model artifacts."""

    # --- sampling / extraction ---
    target_fps: float = 15.0            # frames sampled per second of video
                                        # (15 = native rate of most webcam sources; at the old
                                        # 12 the keep/skip pattern halved effective rate to ~7.5)
    backend: str = "mediapipe"          # "mediapipe" | "mediapipe+l2cs" | "mediapipe+openface3"
    min_face_ratio: float = 0.70        # videos below this get flagged for manual check
    max_num_faces: int = 1

    # --- signal cleaning (Stage 2) ---
    max_gap_seconds: float = 0.5        # interpolate blink/dropout gaps shorter than this
    median_filter_k: int = 3            # odd kernel size for median smoothing (0/1 = off)
    calibrate_gaze: bool = True         # per-video: subtract each person's median gaze
                                        # (baseline) so cross-person offset != signal

    # --- windowing (Stage 2) ---
    window_seconds: float = 20.0
    stride_seconds: float = 10.0
    min_valid_ratio: float = 0.5        # window needs this fraction of face-found frames

    # --- blink / EAR ---
    ear_blink_threshold: float = 0.18   # EAR below this = eye closed (blink)

    # --- saccade detection (velocity based, on normalized gaze) ---
    # Real webcam iris-in-eye gaze has a MUCH smaller amplitude (std ~0.05) than a
    # calibrated eye tracker, so detection is scale-ADAPTIVE by default: the velocity
    # threshold is derived per-signal from its own robust velocity spread. Absolute
    # thresholds below are only used when adaptive_saccade=False.
    adaptive_saccade: bool = True
    saccade_velocity_k: float = 2.5        # thresh = median(|v|) + k * MAD(|v|)
    saccade_velocity_thresh: float = 3.0   # absolute fallback (units/sec)
    min_saccade_amplitude: float = 0.02    # ignore micro-movements below this (norm units)
    min_fixation_seconds: float = 0.10

    # --- reading-sweep detection ---
    # A reading line = a sustained left->right progression of gaze_x terminated by a
    # fast right->left return saccade ("carriage return"). Amplitude thresholds are
    # expressed as a FRACTION of the window's own horizontal gaze range so they work
    # at any gaze scale; the reading rhythm is searched only in a plausible line-cadence
    # band [reading_lag_min_s, autocorr_max_lag_seconds].
    sweep_min_progression_frac: float = 0.25   # L->R travel as fraction of gaze_x range
    return_saccade_min_amp_frac: float = 0.25  # R->L jump as fraction of gaze_x range
    sweep_min_range: float = 0.03              # ignore windows with near-zero gaze motion
    reading_lag_min_s: float = 1.0             # reading lines rarely faster than ~1/s
    autocorr_max_lag_seconds: float = 6.0      # ...and rarely slower than ~1/6s

    # --- gaze grid (spatial occupancy entropy) ---
    grid_n: int = 3                     # 3x3 occupancy grid over (gaze_x, gaze_y)
    off_center_thresh: float = 0.30     # |gaze| beyond this counts as "off-center"

    # --- head aversion ---
    head_averted_yaw_deg: float = 20.0

    # --- speaking detection (v2: visual VAD on the mouth channel) ---
    # Hysteresis thresholds sit at fractions of the video's own activity dynamic
    # range [P10, P90] — scale-free, robust to how much of the video is speech.
    vad_act_window_s: float = 1.0       # rolling-std window for mouth activity
    vad_on_frac: float = 0.30           # ON  = floor + frac * (P90 - floor)
    vad_off_frac: float = 0.15          # OFF = floor + frac * (P90 - floor)
    vad_floor_pctile: float = 10.0      # per-video activity floor percentile
    vad_min_dynamic: float = 2.0        # P90/floor below this = no speech dynamic
    vad_abs_min_act: float = 0.010      # absolute ON floor (MAR-std units): real
                                        # articulation moves the inner-lip ratio far
                                        # more than tracking noise on a still mouth,
                                        # and MAR is scale-free across face sizes
    vad_min_burst_s: float = 0.6        # discard speaking bursts shorter than this
    vad_merge_gap_s: float = 0.5        # merge silent gaps shorter than this
    prespeech_window_s: float = 1.0     # look-back window for pre-speech gaze shift

    # --- sequence model (CNN) ---
    seq_window_seconds: float = 30.0
    seq_resample_fps: float = 12.0      # -> seq_len = seq_window_seconds * seq_resample_fps

    # --- video-level verdict ---
    topk_fraction: float = 0.30         # verdict = mean of top-k window probabilities
    topk_min: int = 3

    # --- CV ---
    n_splits: int = 5
    loo_threshold_videos: int = 15      # if < this many videos/class, use leave-one-video-out
    random_state: int = 42

    def seq_len(self) -> int:
        return int(round(self.seq_window_seconds * self.seq_resample_fps))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        data = json.loads(Path(path).read_text())
        # tolerate configs written by a different version (ignore unknown keys)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_CONFIG = Config()


# --------------------------------------------------------------------------- #
# MediaPipe FaceMesh landmark indices (refine_landmarks=True, 478 points)
# --------------------------------------------------------------------------- #
# Right eye = person's right (appears on the LEFT of the image). Left = mirror.
RIGHT_EYE = dict(outer=33, inner=133, top=159, bottom=145, top2=158, bottom2=153, iris=468)
LEFT_EYE = dict(outer=263, inner=362, top=386, bottom=374, top2=385, bottom2=380, iris=473)

# Head-pose solvePnP correspondences (nose, chin, eye corners, mouth corners).
POSE_LANDMARKS = dict(nose=1, chin=152, left_eye=33, right_eye=263, left_mouth=61, right_mouth=291)

# Mouth landmarks for the inner-lip aspect ratio (visual voice-activity signal):
# 13/14 = upper/lower inner lip midpoints, 61/291 = mouth corners.
MOUTH = dict(top=13, bottom=14, left=61, right=291)

# Canonical 3D face model (mm) matching POSE_LANDMARKS order.
POSE_MODEL_3D = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -330.0, -65.0),   # chin
        (-225.0, 170.0, -135.0),  # left eye outer corner
        (225.0, 170.0, -135.0),   # right eye outer corner
        (-150.0, -150.0, -125.0),  # left mouth corner
        (150.0, -150.0, -125.0),   # right mouth corner
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# Signal utilities
# --------------------------------------------------------------------------- #
def interpolate_gaps(values: np.ndarray, valid: np.ndarray, dt: float,
                     max_gap_seconds: float) -> np.ndarray:
    """Linearly interpolate invalid samples, but only across gaps shorter than
    max_gap_seconds. Longer gaps are left as NaN (real signal loss, not a blink)."""
    values = values.astype(float).copy()
    valid = valid.astype(bool)
    n = len(values)
    if valid.sum() < 2:
        return values
    max_gap = max(1, int(round(max_gap_seconds / max(dt, 1e-6))))

    idx = np.arange(n)
    values[~valid] = np.nan

    # Find contiguous invalid runs; interpolate only short ones bounded by valid samples.
    i = 0
    while i < n:
        if not valid[i]:
            j = i
            while j < n and not valid[j]:
                j += 1
            run = j - i
            has_left = i - 1 >= 0
            has_right = j < n
            if run <= max_gap and has_left and has_right:
                lo, hi = i - 1, j
                values[i:j] = np.interp(idx[i:j], [lo, hi], [values[lo], values[hi]])
            i = j
        else:
            i += 1
    return values


def median_filter_1d(values: np.ndarray, k: int) -> np.ndarray:
    """Simple NaN-aware odd-kernel median filter. k<=1 is a no-op."""
    if k is None or k <= 1:
        return values
    if k % 2 == 0:
        k += 1
    n = len(values)
    half = k // 2
    out = values.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        win = values[lo:hi]
        win = win[~np.isnan(win)]
        if win.size:
            out[i] = np.median(win)
    return out


def resample_uniform(t: np.ndarray, values: np.ndarray, fps: float):
    """Resample an irregular (t, values) series onto a uniform grid at `fps`.
    NaNs are ignored in the interpolation. Returns (t_uniform, values_uniform, dt)."""
    dt = 1.0 / fps
    if len(t) < 2:
        return t, values, dt
    t0, t1 = float(t[0]), float(t[-1])
    n = max(2, int(math.floor((t1 - t0) / dt)) + 1)
    tu = t0 + np.arange(n) * dt
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return tu, np.full(n, np.nan), dt
    vu = np.interp(tu, t[finite], values[finite])
    return tu, vu, dt


# --------------------------------------------------------------------------- #
# Saccade + reading-sweep detection  (operate on cleaned, uniformly sampled gaze)
# --------------------------------------------------------------------------- #
@dataclass
class Saccade:
    start: int
    end: int
    amplitude: float      # signed peak-to-peak of gaze_x across the event
    direction: float      # radians, atan2(dy, dx)
    duration: float


def _mad(x: np.ndarray) -> float:
    """Robust spread: median absolute deviation scaled to ~std."""
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def velocity_threshold(gaze_x: np.ndarray, gaze_y: np.ndarray, dt: float, cfg: Config) -> float:
    """Per-signal saccade velocity threshold. Adaptive (scale-invariant) by default."""
    if not cfg.adaptive_saccade:
        return cfg.saccade_velocity_thresh
    vx = np.gradient(gaze_x, dt)
    vy = np.gradient(gaze_y, dt)
    speed = np.hypot(vx, vy)
    thr = float(np.median(speed) + cfg.saccade_velocity_k * _mad(speed))
    return max(thr, 1e-6)


def detect_saccades(gaze_x: np.ndarray, gaze_y: np.ndarray, dt: float, cfg: Config,
                    vel_thresh: float = None):
    """Velocity-threshold saccade segmentation on 2D gaze. Returns list[Saccade]."""
    n = len(gaze_x)
    if n < 3:
        return []
    if vel_thresh is None:
        vel_thresh = velocity_threshold(gaze_x, gaze_y, dt, cfg)
    vx = np.gradient(gaze_x, dt)
    vy = np.gradient(gaze_y, dt)
    speed = np.hypot(vx, vy)
    is_sacc = speed > vel_thresh

    saccades = []
    i = 0
    while i < n:
        if is_sacc[i]:
            j = i
            while j < n and is_sacc[j]:
                j += 1
            dx = gaze_x[min(j, n - 1)] - gaze_x[i]
            dy = gaze_y[min(j, n - 1)] - gaze_y[i]
            amp = math.hypot(dx, dy)
            if amp >= cfg.min_saccade_amplitude:
                saccades.append(
                    Saccade(i, j, float(dx), math.atan2(dy, dx), (j - i) * dt)
                )
            i = j
        else:
            i += 1
    return saccades


def detect_return_sweeps(gx: np.ndarray, dt: float, cfg: Config, vel_thresh: float):
    """Reading 'carriage returns' on a cleaned gaze_x signal.

    Returns (times, progressions): the time of each fast right->left return jump
    that was preceded by a sufficient left->right reading progression. Amplitude
    thresholds scale with the signal's own robust range. Exposed separately so
    v2 can attribute each sweep event to speaking vs silent periods.
    """
    n = len(gx)
    lo, hi = np.percentile(gx, 5), np.percentile(gx, 95)
    gaze_range = float(hi - lo)
    if gaze_range < cfg.sweep_min_range or n < 5:
        return [], []
    prog_min = cfg.sweep_min_progression_frac * gaze_range
    ret_min = cfg.return_saccade_min_amp_frac * gaze_range
    vx = np.gradient(gx, dt)
    fast_left = vx < -vel_thresh  # return saccade = fast right->left move

    return_times, progressions = [], []
    i, last_return_end = 0, 0
    while i < n:
        if fast_left[i]:
            j = i
            while j < n and fast_left[j]:
                j += 1
            jump = gx[i] - gx[min(j, n - 1)]  # positive = leftward
            if jump >= ret_min:
                seg = gx[last_return_end:i + 1]
                prog = float(seg.max() - seg.min()) if seg.size else 0.0
                if prog >= prog_min:
                    return_times.append(i * dt)
                    progressions.append(prog)
                    last_return_end = j
            i = j
        else:
            i += 1
    return return_times, progressions


def reading_sweep_features(gaze_x: np.ndarray, dt: float, cfg: Config,
                           vel_thresh: float = None) -> dict:
    """Detect the line-by-line reading rhythm on gaze_x.

    Reading an on-screen overlay produces: slow left->right progression (eyes
    tracking a text line) terminated by a fast right->left return saccade
    (carriage return), repeated at a roughly constant cadence.

    Amplitude thresholds scale with the window's own horizontal gaze range, so
    detection works at real webcam gaze amplitudes (std ~0.05) as well as the
    large synthetic scale. Returns counts, lines-per-minute, and the reading-band
    autocorrelation rhythm strength/lag.
    """
    n = len(gaze_x)
    duration = max(n * dt, 1e-6)
    out = dict(
        return_sweep_count=0,
        lines_per_minute=0.0,
        mean_line_progression=0.0,
        sweep_regularity=0.0,
        autocorr_peak=0.0,
        autocorr_lag_s=0.0,
    )
    if n < 5 or not np.isfinite(gaze_x).any():
        return out

    gx = gaze_x.copy()
    gx[~np.isfinite(gx)] = np.nanmean(gx[np.isfinite(gx)]) if np.isfinite(gx).any() else 0.0

    # gaze range (robust: 5th..95th pct) sets the amplitude scale for this window
    lo, hi = np.percentile(gx, 5), np.percentile(gx, 95)
    gaze_range = float(hi - lo)
    if vel_thresh is None:
        vel_thresh = velocity_threshold(gx, np.zeros_like(gx), dt, cfg)

    if gaze_range >= cfg.sweep_min_range:
        return_times, progressions = detect_return_sweeps(gx, dt, cfg, vel_thresh)
        count = len(return_times)
        out["return_sweep_count"] = count
        out["lines_per_minute"] = 60.0 * count / duration
        if progressions:
            out["mean_line_progression"] = float(np.mean(progressions))
        if count >= 3:
            gaps = np.diff(return_times)
            cv = np.std(gaps) / (np.mean(gaps) + 1e-6)
            out["sweep_regularity"] = float(max(0.0, 1.0 - cv))

    # Reading-rhythm autocorrelation, searched only in the line-cadence band.
    peak, lag = _autocorr_peak(gx, dt, cfg.autocorr_max_lag_seconds, cfg.reading_lag_min_s)
    out["autocorr_peak"] = peak
    out["autocorr_lag_s"] = lag
    return out


def _autocorr_peak(x: np.ndarray, dt: float, max_lag_seconds: float, min_lag_seconds: float = 0.0):
    """Return (peak_normalized_autocorr, lag_seconds) over lags in [min_lag, max_lag]."""
    x = x - np.mean(x)
    var = np.dot(x, x)
    if var < 1e-9 or len(x) < 4:
        return 0.0, 0.0
    max_lag = min(len(x) - 1, int(round(max_lag_seconds / max(dt, 1e-6))))
    min_lag = max(1, int(round(min_lag_seconds / max(dt, 1e-6))))
    if max_lag < 2 or min_lag >= max_lag:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:] / var
    seg = ac[min_lag:max_lag + 1]
    k = int(np.argmax(seg))
    return float(seg[k]), float((k + min_lag) * dt)


# --------------------------------------------------------------------------- #
# Gaze-extraction backends
# --------------------------------------------------------------------------- #
class GazeBackend:
    """Interface. process(frame_bgr) -> dict of per-frame channel values."""

    channels: list = []

    def process(self, frame_bgr) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self):
        pass


def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _eye_gaze(landmarks, eye, w, h):
    """Iris center normalized within the eye aperture -> (gx, gy) in ~[-1, 1]."""
    outer = _pt(landmarks, eye["outer"], w, h)
    inner = _pt(landmarks, eye["inner"], w, h)
    top = _pt(landmarks, eye["top"], w, h)
    bottom = _pt(landmarks, eye["bottom"], w, h)
    iris = _pt(landmarks, eye["iris"], w, h)

    cx = (outer[0] + inner[0]) / 2.0
    cy = (top[1] + bottom[1]) / 2.0
    half_w = abs(outer[0] - inner[0]) / 2.0
    half_h = abs(top[1] - bottom[1]) / 2.0
    gx = (iris[0] - cx) / (half_w + 1e-6)
    gy = (iris[1] - cy) / (half_h + 1e-6)
    return gx, gy


def _eye_ear(landmarks, eye, w, h):
    """Eye aspect ratio: (|top1-bot1| + |top2-bot2|) / (2 * |outer-inner|)."""
    outer = _pt(landmarks, eye["outer"], w, h)
    inner = _pt(landmarks, eye["inner"], w, h)
    v1 = np.linalg.norm(_pt(landmarks, eye["top"], w, h) - _pt(landmarks, eye["bottom"], w, h))
    v2 = np.linalg.norm(_pt(landmarks, eye["top2"], w, h) - _pt(landmarks, eye["bottom2"], w, h))
    hd = np.linalg.norm(outer - inner)
    return float((v1 + v2) / (2.0 * hd + 1e-6))


def _mouth_mar(landmarks, w, h):
    """Inner-lip mouth aspect ratio: vertical inner-lip gap / mouth width.
    Scale-invariant; the per-frame input to visual voice-activity detection."""
    top = _pt(landmarks, MOUTH["top"], w, h)
    bottom = _pt(landmarks, MOUTH["bottom"], w, h)
    left = _pt(landmarks, MOUTH["left"], w, h)
    right = _pt(landmarks, MOUTH["right"], w, h)
    width = np.linalg.norm(right - left)
    return float(np.linalg.norm(top - bottom) / (width + 1e-6))


def _head_pose(landmarks, w, h):
    """yaw, pitch, roll (degrees) via solvePnP on 6 canonical points."""
    import cv2

    pts_2d = np.array(
        [
            _pt(landmarks, POSE_LANDMARKS["nose"], w, h),
            _pt(landmarks, POSE_LANDMARKS["chin"], w, h),
            _pt(landmarks, POSE_LANDMARKS["left_eye"], w, h),
            _pt(landmarks, POSE_LANDMARKS["right_eye"], w, h),
            _pt(landmarks, POSE_LANDMARKS["left_mouth"], w, h),
            _pt(landmarks, POSE_LANDMARKS["right_mouth"], w, h),
        ],
        dtype=np.float64,
    )
    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(POSE_MODEL_3D, pts_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
        roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
        yaw = 0.0
        roll = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
    return yaw, pitch, roll


# Official 478-landmark (with iris) FaceLandmarker bundle — same model the
# in-browser @mediapipe/tasks-vision FaceLandmarker uses.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_face_landmarker_model(dest: Optional[Path] = None) -> Path:
    """Download the FaceLandmarker .task bundle to models/ if not already present."""
    dest = dest or (MODELS_DIR / "face_landmarker.task")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model -> {dest} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, dest)
    return dest


def _euler_from_matrix(M):
    """yaw, pitch, roll (degrees) from a 4x4 facial-transformation matrix.
    Uses the rotation submatrix; robust and metric (from MediaPipe), unlike the
    6-point solvePnP fallback which suffers sign flips on webcam frames."""
    R = np.asarray(M)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-R[2, 0], sy))
        yaw = 0.0
        roll = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    return yaw, pitch, roll


class MediaPipeBackend(GazeBackend):
    """Always-on backbone: iris gaze, head pose, EAR/blink.

    Uses the MediaPipe **Tasks** FaceLandmarker (478 landmarks incl. iris), which
    is the unified API across recent mediapipe versions and matches the future
    in-browser deployment target. IMAGE running mode = stateless per-frame detect
    (we already subsample frames, so cross-frame tracking adds little)."""

    channels = BASE_CHANNELS

    def __init__(self, cfg: Config = DEFAULT_CONFIG, model_path: Optional[Path] = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.cfg = cfg
        self._mp = mp
        model_path = model_path or ensure_face_landmarker_model()
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=cfg.max_num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,  # metric head pose (stable)
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame_bgr):
        """Return (landmarks, xform_matrix_or_None, w, h) or (None, None, w, h).
        Landmarks are NormalizedLandmark objects with .x/.y in [0, 1]."""
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(rgb))
        res = self._landmarker.detect(mp_image)
        if not res.face_landmarks:
            return None, None, w, h
        xform = None
        if getattr(res, "facial_transformation_matrixes", None):
            xform = np.asarray(res.facial_transformation_matrixes[0]).reshape(4, 4)
        return res.face_landmarks[0], xform, w, h

    def process(self, frame_bgr) -> dict:
        landmarks, xform, w, h = self.detect(frame_bgr)
        if landmarks is None:
            return dict(face_found=0, gaze_x=np.nan, gaze_y=np.nan,
                        yaw=np.nan, pitch=np.nan, roll=np.nan, ear=np.nan,
                        mouth=np.nan, blink=0)
        gxr, gyr = _eye_gaze(landmarks, RIGHT_EYE, w, h)
        gxl, gyl = _eye_gaze(landmarks, LEFT_EYE, w, h)
        gaze_x = float(np.clip((gxr + gxl) / 2.0, -2.0, 2.0))
        gaze_y = float(np.clip((gyr + gyl) / 2.0, -2.0, 2.0))
        ear = (_eye_ear(landmarks, RIGHT_EYE, w, h) + _eye_ear(landmarks, LEFT_EYE, w, h)) / 2.0
        mouth = _mouth_mar(landmarks, w, h)
        if xform is not None:
            yaw, pitch, roll = _euler_from_matrix(xform)
        else:
            yaw, pitch, roll = _head_pose(landmarks, w, h)
        blink = 1 if ear < self.cfg.ear_blink_threshold else 0
        return dict(face_found=1, gaze_x=gaze_x, gaze_y=gaze_y,
                    yaw=yaw, pitch=pitch, roll=roll, ear=ear, mouth=mouth, blink=blink)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


class L2CSBackend(GazeBackend):
    """Appearance-based gaze (yaw/pitch in camera frame) via L2CS-Net.

    Requires `pip install l2cs`. Runs on MPS/CPU. Reuses the MediaPipe face bbox
    when available to avoid a second face detector.
    """

    channels = L2CS_CHANNELS

    def __init__(self, cfg: Config = DEFAULT_CONFIG, device: Optional[str] = None):
        from l2cs import Pipeline  # type: ignore
        import torch

        self.cfg = cfg
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.pipeline = Pipeline(weights=None, arch="ResNet50", device=self.device)

    def process(self, frame_bgr) -> dict:
        try:
            res = self.pipeline.step(frame_bgr)
            yaw = float(np.mean(res.yaw)) if len(res.yaw) else np.nan
            pitch = float(np.mean(res.pitch)) if len(res.pitch) else np.nan
        except Exception:
            yaw = pitch = np.nan
        return dict(gaze_yaw_l2cs=yaw, gaze_pitch_l2cs=pitch)


class OpenFace3Backend(GazeBackend):
    """Alternative appearance backend (OpenFace 3.0). Heavier multi-model install.
    Wired as a stub that raises a clear message until the user installs it; the
    channel names match L2CS so the rest of the pipeline is backend-agnostic."""

    channels = L2CS_CHANNELS

    def __init__(self, cfg: Config = DEFAULT_CONFIG, device: Optional[str] = None):
        raise NotImplementedError(
            "OpenFace 3.0 backend is not installed. Install from "
            "https://github.com/CMU-MultiComp-Lab/OpenFace-3.0 and implement "
            "process() to return {'gaze_yaw_l2cs': .., 'gaze_pitch_l2cs': ..}. "
            "The bake-off in train.py treats it as an alternative appearance backend."
        )


# Imported here rather than at the top so the line numbers frontend/js cites into
# this file (MediaPipeBackend and earlier) stay put.
import threading  # noqa: E402


class _MediaPipePool:
    """Worker threads, each owning one FaceLandmarker (created lazily on first use)."""

    def __init__(self, cfg: Config, workers: int):
        from concurrent.futures import ThreadPoolExecutor
        self.cfg = cfg
        self.workers = workers
        self._local = threading.local()
        self._lock = threading.Lock()
        self._backends = []
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mediapipe")

    def _backend(self) -> "MediaPipeBackend":
        backend = getattr(self._local, "backend", None)
        if backend is None:
            backend = self._local.backend = MediaPipeBackend(self.cfg)
            with self._lock:
                self._backends.append(backend)
        return backend

    def run(self, frame_bgr) -> dict:
        return self._backend().process(frame_bgr)

    def warm(self):
        """Create every thread's FaceLandmarker now. The barrier holds each task until
        all have started, which forces them onto distinct threads."""
        barrier = threading.Barrier(self.workers)

        def init(_):
            self._backend()
            barrier.wait(timeout=120)

        list(self.executor.map(init, range(self.workers)))

    def close(self):
        # Close explicitly: a FaceLandmarker left for interpreter shutdown to collect
        # raises from its __del__, because MediaPipe's own dispatcher is gone by then.
        self.executor.shutdown(wait=True, cancel_futures=True)
        for backend in self._backends:
            backend.close()
        self._backends = []


# Long-lived pools created by warm_mediapipe(), keyed by (workers, config). A server
# warms one at start-up so the ~0.7 s per-FaceLandmarker creation cost is paid once
# instead of per video; one-off runs (the CLI) use a private pool closed at the end.
_WARM_POOLS = {}
_WARM_POOLS_LOCK = threading.Lock()


def warm_mediapipe(cfg: Config = DEFAULT_CONFIG, workers: int = 2):
    """Create (or reuse) a long-lived MediaPipe thread pool and warm every thread."""
    key = (workers, repr(cfg))
    with _WARM_POOLS_LOCK:
        pool = _WARM_POOLS.get(key) or _WARM_POOLS.setdefault(key, _MediaPipePool(cfg, workers))
    pool.warm()


def close_mediapipe():
    """Close every pool warm_mediapipe() created (call on server shutdown)."""
    with _WARM_POOLS_LOCK:
        pools = list(_WARM_POOLS.values())
        _WARM_POOLS.clear()
    for pool in pools:
        pool.close()


class ParallelMediaPipeBackend:
    """MediaPipeBackend fanned out over a thread pool, one FaceLandmarker per thread.

    IMAGE running mode is stateless per frame, so every frame's output is identical
    to the sequential backend's — only throughput changes (MediaPipe releases the GIL
    while it runs; ~4x on a 10-core M5 at 9 threads). process() hands back the frame's
    record dict straight away and fills it in later, so extract_one's loop is
    unchanged; close() waits for every frame still in flight. At most 4 frames per
    thread are queued, so decoding can never run ahead of inference and fill memory.
    Uses the warm pool from warm_mediapipe() when one exists, else a private pool."""

    channels = BASE_CHANNELS

    def __init__(self, cfg: Config = DEFAULT_CONFIG, workers: int = 2):
        self.cfg = cfg
        with _WARM_POOLS_LOCK:
            self._pool = _WARM_POOLS.get((workers, repr(cfg)))
        self._owns_pool = self._pool is None
        if self._owns_pool:
            self._pool = _MediaPipePool(cfg, workers)
        self._slots = threading.BoundedSemaphore(4 * workers)
        self._pending = []  # (record, future), in submission (= frame) order

    def _run(self, frame_bgr) -> dict:
        try:
            return self._pool.run(frame_bgr)
        finally:
            self._slots.release()

    def process(self, frame_bgr) -> dict:
        self._slots.acquire()
        rec = {}
        self._pending.append((rec, self._pool.executor.submit(self._run, frame_bgr)))
        return rec

    def close(self):
        """Wait for this video's frames; shut a private pool down (a warm one stays up)."""
        pending, self._pending = self._pending, []
        try:
            for rec, fut in pending:
                rec.update(fut.result())
        finally:
            if self._owns_pool:
                self._pool.close()


def make_backends(backend: str, cfg: Config = DEFAULT_CONFIG, workers: int = 1):
    """Return (mediapipe_backend, appearance_backend_or_None) for a backend string.
    workers > 1 runs MediaPipe on that many threads (MediaPipe-only backend)."""
    backend = backend.lower()
    if backend == "mediapipe" and workers > 1:
        return ParallelMediaPipeBackend(cfg, workers), None
    mp_backend = MediaPipeBackend(cfg)
    if backend == "mediapipe":
        return mp_backend, None
    if backend == "mediapipe+l2cs":
        return mp_backend, L2CSBackend(cfg)
    if backend == "mediapipe+openface3":
        return mp_backend, OpenFace3Backend(cfg)
    raise ValueError(f"Unknown backend '{backend}'. Use mediapipe | mediapipe+l2cs | mediapipe+openface3")


def channels_for_backend(backend: str) -> list:
    return BASE_CHANNELS + (L2CS_CHANNELS if backend.lower() != "mediapipe" else [])


# --------------------------------------------------------------------------- #
# Synthetic signals + selftest
# --------------------------------------------------------------------------- #
def synth_reading_signal(duration_s=60.0, fps=12.0, line_seconds=2.5, seed=0):
    """Sawtooth reading gaze: slow L->R ramp then fast return, at a steady cadence."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    dt = 1.0 / fps
    t = np.arange(n) * dt
    phase = (t % line_seconds) / line_seconds          # 0..1 within a line
    gaze_x = -0.8 + 1.6 * phase                          # ramp -0.8 -> +0.8
    gaze_x += rng.normal(0, 0.02, n)                     # small tremor
    gaze_y = 0.05 * np.sin(2 * np.pi * t / (line_seconds * 3)) + rng.normal(0, 0.02, n)
    return t, gaze_x, gaze_y, dt


def synth_random_walk(duration_s=60.0, fps=12.0, seed=1):
    """Aimless bounded random-walk gaze (no reading rhythm)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * fps)
    dt = 1.0 / fps
    t = np.arange(n) * dt
    gaze_x = np.clip(np.cumsum(rng.normal(0, 0.05, n)), -1, 1)
    gaze_y = np.clip(np.cumsum(rng.normal(0, 0.05, n)), -1, 1)
    return t, gaze_x, gaze_y, dt


def run_selftest() -> bool:
    """Verify the sweep detector fires on reading and stays quiet on a random walk."""
    cfg = DEFAULT_CONFIG
    ok = True

    t, gx, gy, dt = synth_reading_signal(seed=0)
    reading = reading_sweep_features(gx, dt, cfg)
    print("[selftest] reading signal :", {k: round(v, 3) for k, v in reading.items()})

    t2, gx2, gy2, dt2 = synth_random_walk(seed=1)
    walk = reading_sweep_features(gx2, dt2, cfg)
    print("[selftest] random walk    :", {k: round(v, 3) for k, v in walk.items()})

    # Assertions: reading must show many, regular sweeps with strong rhythm.
    checks = [
        ("reading return_sweep_count >= 10", reading["return_sweep_count"] >= 10),
        ("reading lines_per_minute >= 15", reading["lines_per_minute"] >= 15),
        ("reading autocorr_peak >= 0.4", reading["autocorr_peak"] >= 0.4),
        ("reading sweeps > 3x walk sweeps",
         reading["return_sweep_count"] > 3 * max(1, walk["return_sweep_count"])),
        ("walk return_sweep_count small", walk["return_sweep_count"] <= reading["return_sweep_count"] // 3 + 1),
    ]
    # Saccade detector sanity on reading signal.
    sacc = detect_saccades(gx, gy, dt, cfg)
    checks.append(("reading has saccades (returns)", len(sacc) >= 10))

    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("[selftest]", "ALL PASS ✅" if ok else "FAILURES ❌")
    return ok


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(0 if run_selftest() else 1)
    print("common.py — import this module, or run with --selftest")
