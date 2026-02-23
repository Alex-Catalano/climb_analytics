#(venv) alex@Alexs-MacBook-Pro climb_analytics % python3 analyze_clip_movenet_video.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264   

# analyze_clip_movenet_video_v2_jump_better.py
# Climbing pose + movement + micro-pause detection + CSV/plot + telemetry video
#
# Improvements focused on climbing + dynos/jumps:
#   1) Crop tracking (for wide shots / small climber)  ✅
#   2) Startup "lock-on" so we don't latch onto a bad bbox at the beginning ✅
#   3) Dynamic crop padding during fast movement (keeps limbs inside crop on dynos) ✅
#   4) Adaptive telemetry stride: process more frames only when moving fast ✅
#   5) Bone-length stabilization (reduces "arm shortens") ✅
#   6) Bbox update gating (prevents crop from teleporting to nonsense) ✅
#   7) Optional MoveNet Thunder toggle (better accuracy, slower) ✅
#
# Notes:
# - This does NOT modify your original video.
# - Proxy conversion is still supported.
# - The system does NOT "revert to previous pose" as a primary strategy.
#   It instead improves acquisition + crop + constraints so pose stays correct during jumps.

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import shutil


# -----------------------------
# Tunables
# -----------------------------
MOVEMENT_THRESHOLD = 0.035
MIN_PAUSE_SECONDS = 1.25
MERGE_GAP_SECONDS = 0.35

# Analysis sampling
FRAME_STRIDE = 3  # keep analysis consistent + fast; telemetry uses adaptive stride

# Model paths (Lightning is fast, Thunder is more accurate)
MODEL_LIGHTNING = "movenet_singlepose_lightning.tflite"
MODEL_THUNDER = "movenet_singlepose_thunder.tflite"
USE_THUNDER = False  # can be overridden by CLI

MODEL_PATH = str(Path(__file__).resolve().parent / (MODEL_THUNDER if USE_THUNDER else MODEL_LIGHTNING))

CONF_THRESHOLD = 0.03
MIN_GOOD_KPTS = 5
MIN_OVERLAP_KPTS = 6

DEBUG = False


# -----------------------------
# Telemetry speed controls
# -----------------------------
WRITE_TELEMETRY_AT_PROCESSED_FPS = True
PRE_DOWNSCALE_MAX_DIM = 1280

# Adaptive telemetry stride:
TELEMETRY_STRIDE_SLOW = 4   # default
TELEMETRY_STRIDE_FAST = 1   # during dynos / fast movement
FAST_STRIDE_EMA_THR = 0.055  # if movement EMA exceeds -> stride FAST

# Movement EMA (used for adaptive stride + dynamic crop)
MOVEMENT_EMA_ALPHA = 0.25


# -----------------------------
# Pose stability controls
# -----------------------------
POSE_SMOOTH_ALPHA = 0.30
FREEZE_CONF_THR = 0.20
MAX_JOINT_JUMP = 0.06
MAX_JOINT_JUMP_FAST = 0.10
FAST_MOVE_MOVEMENT_EMA_THR = 0.060  # if movement EMA above this, allow more jump


# Draw thresholds (stricter distal joints)
DRAW_THR_BY_JOINT = np.array([
    0.20, 0.20, 0.20, 0.20, 0.20,   # face
    0.15, 0.15,                     # shoulders
    0.20, 0.20,                     # elbows
    0.35, 0.35,                     # wrists
    0.12, 0.12,                     # hips
    0.18, 0.18,                     # knees
    0.30, 0.30                      # ankles
], dtype=np.float32)

DRAW_CONF_THRESHOLD = 0.12
MAX_EDGE_LEN_NORM = 0.45


# -----------------------------
# Proxy conversion
# -----------------------------
AUTO_PROXY = True
PROXY_WIDTH = 1280
PROXY_FPS = 30


# -----------------------------
# Crop tracking (wide shots)
# -----------------------------
USE_CROP_TRACKING = True
CROP_SMOOTH = 0.25
CROP_MIN_SIZE = 320
CROP_ASPECT_SQUARE = True
CROP_FALLBACK_FULL_FRAMES = 14

# dynamic padding
CROP_PAD_SLOW = 0.35
CROP_PAD_FAST = 0.60
FAST_PAD_EMA_THR = 0.055

# bbox build conf threshold
BBOX_CONF_THR = 0.12

# bbox update gating (prevents crop teleporting)
BBOX_IOU_MIN = 0.10
BBOX_AREA_CHANGE_MAX = 3.5


# -----------------------------
# Startup lock-on (fix "bad skeleton at start")
# -----------------------------
CORE_JOINTS = [5, 6, 11, 12]  # shoulders + hips
STARTUP_FULLFRAME_SECONDS = 0.8  # force full-frame inference initially
LOCKON_CORE_CONF = 0.18
LOCKON_GOOD_FRAMES = 4


# -----------------------------
# Bone-length stabilization (reduce "arm shortens")
# -----------------------------
BONES = [
    (5, 7), (7, 9),    # left upper arm, left forearm
    (6, 8), (8, 10),   # right upper arm, right forearm
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
BONE_REF_UPDATE_ALPHA = 0.10
BONE_STAB_STRENGTH = 0.35
BONE_REF_MIN_CONF = 0.25
BONE_STAB_DISTAL_CONF_THR = 0.35


# -----------------------
# ffmpeg helpers
# -----------------------
def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ffmpeg_make_proxy(in_path: str, out_path: str, width=1280, fps=30):
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    vf = f"scale={int(width)}:-2,fps={int(fps)},format=yuv420p"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "20",
        "-preset", "veryfast",
        "-movflags", "+faststart",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def ffmpeg_reencode_to_h264_qt(in_path: str, out_path: str, fps: int | None = None, max_h: int | None = 1920):
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    vf_parts = []
    if max_h is not None:
        vf_parts.append(f"scale=-2:{int(max_h)}")
    if fps is not None:
        vf_parts.append(f"fps={int(fps)}")
    vf_parts.append("format=yuv420p")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_path),
        "-vf", vf,
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level", "4.1",
        "-crf", "20",
        "-preset", "veryfast",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-movflags", "+faststart",
        "-video_track_timescale", "600",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# -----------------------
# Video / frame utilities
# -----------------------
def open_video_or_raise(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def raw_frames(cap: cv2.VideoCapture):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def maybe_downscale_for_movenet(frame_bgr: np.ndarray, max_dim: int):
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return frame_bgr
    scale = max_dim / float(m)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


# -----------------------
# Crop helpers
# -----------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    a_area = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
    b_area = float(max(1, bx2 - bx1) * max(1, by2 - by1))
    union = max(1e-9, a_area + b_area - inter)
    return inter / union


def bbox_area(bb):
    if bb is None:
        return 0.0
    x1, y1, x2, y2 = bb
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def bbox_from_keypoints_fullframe(kpts_yx_conf, frame_w, frame_h, conf_thr=0.20):
    ys = kpts_yx_conf[:, 0]
    xs = kpts_yx_conf[:, 1]
    cs = kpts_yx_conf[:, 2]
    good = cs >= conf_thr
    if int(good.sum()) < 4:
        return None

    x1 = float(xs[good].min()) * frame_w
    x2 = float(xs[good].max()) * frame_w
    y1 = float(ys[good].min()) * frame_h
    y2 = float(ys[good].max()) * frame_h

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    return (x1, y1, x2, y2)


def expand_and_square_bbox(x1, y1, x2, y2, frame_w, frame_h, pad=0.35, min_size=320, make_square=True):
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5

    w *= (1.0 + 2.0 * pad)
    h *= (1.0 + 2.0 * pad)

    if make_square:
        s = max(w, h)
        w = h = s

    w = max(w, float(min_size))
    h = max(h, float(min_size))

    nx1 = cx - w * 0.5
    nx2 = cx + w * 0.5
    ny1 = cy - h * 0.5
    ny2 = cy + h * 0.5

    nx1 = clamp(nx1, 0, frame_w - 2)
    ny1 = clamp(ny1, 0, frame_h - 2)
    nx2 = clamp(nx2, nx1 + 2, frame_w - 1)
    ny2 = clamp(ny2, ny1 + 2, frame_h - 1)
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def smooth_bbox(prev, curr, alpha=0.25):
    if prev is None:
        return curr
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr
    sx1 = int(round(alpha * cx1 + (1 - alpha) * px1))
    sy1 = int(round(alpha * cy1 + (1 - alpha) * py1))
    sx2 = int(round(alpha * cx2 + (1 - alpha) * px2))
    sy2 = int(round(alpha * cy2 + (1 - alpha) * py2))
    return (sx1, sy1, sx2, sy2)


def crop_frame(frame_bgr, bbox):
    x1, y1, x2, y2 = bbox
    return frame_bgr[y1:y2, x1:x2].copy()


def kpts_crop_to_fullnorm(kpts_crop, bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox
    cw = max(1, x2 - x1)
    ch = max(1, y2 - y1)
    out = kpts_crop.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] * ch + y1) / float(frame_h)
    out[:, 1] = (out[:, 1] * cw + x1) / float(frame_w)
    return out


# -----------------------
# Bone helpers
# -----------------------
def update_bone_ref(bone_ref, yx, conf, alpha=BONE_REF_UPDATE_ALPHA):
    if bone_ref is None:
        bone_ref = {}
    for a, b in BONES:
        if float(conf[a]) >= BONE_REF_MIN_CONF and float(conf[b]) >= BONE_REF_MIN_CONF:
            d = float(np.linalg.norm(yx[b] - yx[a]))
            if d > 1e-6:
                key = (a, b)
                if key not in bone_ref:
                    bone_ref[key] = d
                else:
                    bone_ref[key] = (1 - alpha) * bone_ref[key] + alpha * d
    return bone_ref


def stabilize_bones(yx, conf, bone_ref, strength=BONE_STAB_STRENGTH):
    if bone_ref is None:
        return yx
    out = yx.copy()
    for a, b in BONES:
        key = (a, b)
        if key not in bone_ref:
            continue

        # adjust distal joint when it's low confidence
        if float(conf[b]) < BONE_STAB_DISTAL_CONF_THR and float(conf[a]) > 0.20:
            ref = float(bone_ref[key])
            v = out[b] - out[a]
            d = float(np.linalg.norm(v))
            if d > 1e-6 and ref > 1e-6:
                target = out[a] + (v / d) * ref
                out[b] = (1 - strength) * out[b] + strength * target
    return out


# -----------------------
# MoveNet
# -----------------------
def load_interpreter(model_path: str):
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}\n"
            f"Put the model next to this script.\n"
            f"Lightning: {MODEL_LIGHTNING}\n"
            f"Thunder:   {MODEL_THUNDER}"
        )

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter, interpreter.get_input_details(), interpreter.get_output_details()


def run_movenet(interpreter, input_details, output_details, frame_bgr):
    frame_bgr = maybe_downscale_for_movenet(frame_bgr, PRE_DOWNSCALE_MAX_DIM)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]["index"])
    return out[0, 0, :, :]  # (17,3)


# -----------------------
# Pause segmentation
# -----------------------
def _segments_from_boolean(times, is_paused):
    segs = []
    in_pause = False
    start = None
    for t, p in zip(times, is_paused):
        if p and not in_pause:
            in_pause = True
            start = t
        elif (not p) and in_pause:
            in_pause = False
            end = t
            if start is not None:
                segs.append((float(start), float(end)))
    if in_pause and start is not None and len(times):
        segs.append((float(start), float(times[-1])))
    return segs


def _filter_and_merge_segments(segs, min_len_s, merge_gap_s):
    segs = [(s, e) for (s, e) in segs if (e - s) >= min_len_s]
    if not segs:
        return []
    segs.sort(key=lambda x: x[0])
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe <= merge_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


# -----------------------
# Visualization (pose)
# -----------------------
SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def _to_pixel_xy(kpts_yx, w, h):
    ys = (kpts_yx[:, 0] * h).astype(int)
    xs = (kpts_yx[:, 1] * w).astype(int)
    return xs, ys


def draw_pose(frame_bgr, kpts, conf_thr=0.03):
    h, w = frame_bgr.shape[:2]
    conf = kpts[:, 2].astype(np.float32)
    pts_yx = kpts[:, :2].astype(np.float32)

    thr = np.maximum(DRAW_THR_BY_JOINT, float(conf_thr))
    good = conf >= thr

    xs, ys = _to_pixel_xy(pts_yx, w, h)

    for a, b in SKELETON_EDGES:
        if good[a] and good[b]:
            dy = float(pts_yx[a, 0] - pts_yx[b, 0])
            dx = float(pts_yx[a, 1] - pts_yx[b, 1])
            d = (dx * dx + dy * dy) ** 0.5
            if d <= MAX_EDGE_LEN_NORM:
                cv2.line(frame_bgr, (xs[a], ys[a]), (xs[b], ys[b]), (0, 255, 0), 2, cv2.LINE_AA)

    for i in range(17):
        if good[i]:
            cv2.circle(frame_bgr, (xs[i], ys[i]), 3, (0, 255, 255), -1, cv2.LINE_AA)

    return good


def draw_panel(panel, lines, header="Telemetry"):
    panel[:] = (20, 20, 20)
    cv2.putText(panel, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (240, 240, 240), 2, cv2.LINE_AA)
    y = 60
    for txt, color in lines:
        cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
        y += 26


def draw_sparkline(panel, values, vmin=0.0, vmax=0.12, x0=12, y0=250, w=260, h=120, thresh=None):
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    if len(values) < 2:
        return
    vals = np.clip(np.array(values, dtype=np.float32), vmin, vmax)
    xs = np.linspace(x0, x0 + w - 1, len(vals)).astype(int)
    ys = (y0 + h - 1 - (vals - vmin) / (vmax - vmin + 1e-9) * (h - 2)).astype(int)
    for i in range(1, len(vals)):
        cv2.line(panel, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                 (200, 200, 200), 2, cv2.LINE_AA)
    if thresh is not None:
        t = float(np.clip(thresh, vmin, vmax))
        yth = int(y0 + h - 1 - (t - vmin) / (vmax - vmin + 1e-9) * (h - 2))
        cv2.line(panel, (x0, yth), (x0 + w, yth), (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "thr", (x0 + w - 45, yth - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)


# -----------------------
# Pose smoothing (joint-wise)
# -----------------------
def smooth_pose(prev_yx, prev_conf, curr_yx, curr_conf, alpha, max_jump, freeze_thr):
    if prev_yx is None or prev_conf is None:
        return curr_yx, curr_conf

    out = curr_yx.copy()

    low = curr_conf < freeze_thr
    out[low] = prev_yx[low]

    delta = out - prev_yx
    delta = np.clip(delta, -max_jump, max_jump)
    out = prev_yx + delta

    smoothed = alpha * out + (1 - alpha) * prev_yx
    return smoothed, curr_conf


def core_mean_conf(kpts):
    return float(np.mean([float(kpts[i, 2]) for i in CORE_JOINTS]))


# -----------------------
# Proxy selection
# -----------------------
def make_or_use_proxy(video_path: str, out_dir: Path) -> str:
    if not AUTO_PROXY:
        return video_path

    p = Path(video_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = p.suffix.lower()
    needs_proxy = ext in [".mov", ".m4v"]

    try:
        cap = open_video_or_raise(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        cap.release()
        if max(w, h) > 1920 or fps > 40:
            needs_proxy = True
    except Exception:
        pass

    if not needs_proxy:
        return video_path

    proxy_path = out_dir / f"{p.stem}_proxy_{PROXY_WIDTH}w_{PROXY_FPS}fps.mp4"
    if proxy_path.exists():
        return str(proxy_path)

    print(f"[proxy] Creating proxy: {proxy_path.name}")
    ffmpeg_make_proxy(str(p), str(proxy_path), width=PROXY_WIDTH, fps=PROXY_FPS)
    return str(proxy_path)


# -----------------------
# Core: crop-tracked inference with lock-on + bbox gating + dynamic padding
# -----------------------
def infer_pose_fullnorm(
    interpreter, input_details, output_details,
    frame_bgr,
    t_seconds: float,
    movement_ema: float,
    bbox_state, bbox_keep,
    lockon_count: int
):
    """
    Returns:
      kpts_fullnorm (17,3)
      bbox_state, bbox_keep
      lockon_count
      debug dict
    """
    H, W = frame_bgr.shape[:2]
    dbg = {"used_crop": False, "pad": 0.0, "bbox_updated": False, "lockon": lockon_count}

    # force full-frame at clip start until lock-on achieved
    in_startup = (t_seconds < STARTUP_FULLFRAME_SECONDS) or (lockon_count < LOCKON_GOOD_FRAMES)
    pad = CROP_PAD_FAST if movement_ema >= FAST_PAD_EMA_THR else CROP_PAD_SLOW
    dbg["pad"] = pad

    kpts = None

    if USE_CROP_TRACKING and (not in_startup) and bbox_state is not None and bbox_keep > 0:
        crop = crop_frame(frame_bgr, bbox_state)
        kpts_crop = run_movenet(interpreter, input_details, output_details, crop)
        kpts = kpts_crop_to_fullnorm(kpts_crop, bbox_state, W, H)
        dbg["used_crop"] = True
        bbox_keep -= 1

        # propose bbox update from pose
        bb0 = bbox_from_keypoints_fullframe(kpts, W, H, conf_thr=BBOX_CONF_THR)
        if bb0 is not None:
            bb1 = expand_and_square_bbox(*bb0, W, H, pad=pad, min_size=CROP_MIN_SIZE, make_square=CROP_ASPECT_SQUARE)

            iou = bbox_iou(bbox_state, bb1)
            a0 = bbox_area(bbox_state)
            a1 = bbox_area(bb1)
            area_mult = (a1 / max(1e-9, a0)) if a0 > 1 else 1.0

            if (iou >= BBOX_IOU_MIN) and (area_mult <= BBOX_AREA_CHANGE_MAX):
                bbox_state = smooth_bbox(bbox_state, bb1, alpha=CROP_SMOOTH)
                bbox_keep = CROP_FALLBACK_FULL_FRAMES
                dbg["bbox_updated"] = True

    if kpts is None:
        # full-frame inference (startup and fallback)
        kpts = run_movenet(interpreter, input_details, output_details, frame_bgr)

    # update lock-on using full-frame pose quality
    if in_startup:
        if core_mean_conf(kpts) >= LOCKON_CORE_CONF:
            lockon_count += 1
        else:
            lockon_count = 0
        dbg["lockon"] = lockon_count

    # bbox acquisition/update from full-frame when in startup OR when bbox missing
    if USE_CROP_TRACKING and (bbox_state is None or in_startup):
        bb0 = bbox_from_keypoints_fullframe(kpts, W, H, conf_thr=BBOX_CONF_THR)
        if bb0 is not None and lockon_count >= LOCKON_GOOD_FRAMES:
            bb1 = expand_and_square_bbox(*bb0, W, H, pad=pad, min_size=CROP_MIN_SIZE, make_square=CROP_ASPECT_SQUARE)
            bbox_state = smooth_bbox(bbox_state, bb1, alpha=CROP_SMOOTH)
            bbox_keep = CROP_FALLBACK_FULL_FRAMES
            dbg["bbox_updated"] = True

    return kpts, bbox_state, bbox_keep, lockon_count, dbg


# -----------------------
# Analysis
# -----------------------
def analyze_video(video_path: str, interpreter, input_details, output_details):
    cap = open_video_or_raise(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    prev_pts = None
    prev_good = None

    bbox_state = None
    bbox_keep = 0
    lockon_count = 0

    movement_scores = []
    times = []
    has_pose = []

    movement_ema = 0.0
    frame_idx = 0

    for frame in raw_frames(cap):
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps

        kpts, bbox_state, bbox_keep, lockon_count, _dbg = infer_pose_fullnorm(
            interpreter, input_details, output_details,
            frame, t, movement_ema,
            bbox_state, bbox_keep,
            lockon_count
        )

        if DEBUG and len(times) == 0:
            conf0 = kpts[:, 2]
            print("conf min/mean/max:", float(conf0.min()), float(conf0.mean()), float(conf0.max()))

        conf = kpts[:, 2].astype(np.float32)
        pts = kpts[:, :2].astype(np.float32)
        good = conf > CONF_THRESHOLD

        score = 0.0
        if good.sum() >= MIN_GOOD_KPTS:
            # update bone ref (only on stable-ish frames)
            # note: analysis doesn't draw, but better pts improves movement stability
            # (bone ref not used here to keep analysis simple + fast)
            if prev_pts is not None and prev_good is not None:
                both = good & prev_good
                if both.sum() >= MIN_OVERLAP_KPTS:
                    diffs = pts[both] - prev_pts[both]
                    score = float(np.linalg.norm(diffs, axis=1).mean())
            prev_pts = pts
            prev_good = good
            has_pose.append(True)
        else:
            prev_pts = None
            prev_good = None
            has_pose.append(False)

        # update EMA after computing score
        movement_ema = MOVEMENT_EMA_ALPHA * score + (1 - MOVEMENT_EMA_ALPHA) * movement_ema

        times.append(float(t))
        movement_scores.append(float(score))
        frame_idx += 1

    cap.release()

    df = pd.DataFrame({"t": times, "movement": movement_scores, "has_pose": has_pose})
    df["is_moving"] = df["movement"] >= MOVEMENT_THRESHOLD
    df["is_paused"] = ~df["is_moving"]

    duration = float(df["t"].iloc[-1]) if len(df) else 0.0

    raw_pause_segments = _segments_from_boolean(df["t"].values, df["is_paused"].values) if len(df) else []
    pause_segments = _filter_and_merge_segments(raw_pause_segments, MIN_PAUSE_SECONDS, MERGE_GAP_SECONDS)

    pause_time = float(sum(e - s for s, e in pause_segments))
    pause_count = int(len(pause_segments))
    move_time = float(max(0.0, duration - pause_time))

    avg_movement = float(df["movement"].mean()) if len(df) else 0.0
    movement_rate = float(df["movement"].sum() / duration) if duration > 0 else 0.0
    pose_detect_rate = float(df["has_pose"].mean()) if len(df) else 0.0

    summary = {
        "video": Path(video_path).name,
        "duration_s": round(duration, 2),
        "micro_pause_time_s": round(pause_time, 2),
        "micro_pause_count": pause_count,
        "move_time_s": round(move_time, 2),
        "avg_movement": round(avg_movement, 6),
        "movement_rate": round(movement_rate, 6),
        "pose_detect_rate": round(pose_detect_rate, 3),
        "used_fps": round(float(fps / FRAME_STRIDE), 2),
    }

    return summary, df, pause_segments, fps


def save_outputs(video_path: str, df: pd.DataFrame, pause_segments, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    csv_path = out_dir / f"{stem}_movement.csv"
    png_path = out_dir / f"{stem}_movement.png"

    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["t"], df["movement"])
    plt.axhline(MOVEMENT_THRESHOLD, linestyle="--")
    for s, e in pause_segments:
        plt.axvspan(s, e, alpha=0.25)
    plt.xlabel("Time (s)")
    plt.ylabel("Movement score (avg joint delta)")
    plt.title(f"{stem} movement + micro-pauses")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return csv_path, png_path


# -----------------------
# Telemetry video (adaptive stride + bone stabilization)
# -----------------------
def write_annotated_video(video_path: str, out_path: str, interpreter, input_details, output_details, panel_w: int = 320, draw_crop_box: bool = False):
    cap0 = open_video_or_raise(video_path)
    fps_in = cap0.get(cv2.CAP_PROP_FPS)
    if not fps_in or fps_in <= 1:
        fps_in = 30.0
    ok, first = cap0.read()
    if not ok:
        cap0.release()
        raise RuntimeError("Could not read first frame.")
    h, w = first.shape[:2]
    cap0.release()

    cap = open_video_or_raise(video_path)

    out_w = w + panel_w
    out_h = h

    # fps_out is approximate when using adaptive stride; this keeps video playable
    # We'll set fps_out based on slow stride (most of the clip), which is fine visually.
    fps_out = float(fps_in / max(1, TELEMETRY_STRIDE_SLOW)) if WRITE_TELEMETRY_AT_PROCESSED_FPS else float(fps_in)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter at: {out_path}")

    # movement tracking
    prev_pts = None
    prev_good = None
    movement_ema = 0.0
    last_score = 0.0
    last_has_pose = False

    # pose smoothing state
    smoothed_yx = None
    smoothed_conf = None

    # crop tracking state
    bbox_state = None
    bbox_keep = 0
    lockon_count = 0

    # bone state
    bone_ref = None

    # micro-pause state
    pause_started_t = None
    confirmed_pause = False
    micro_pause_time = 0.0
    micro_pause_count = 0

    history = []
    window_len = max(5, int(12.0 * (fps_out if fps_out > 0 else 10)))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps_in

        stride_now = TELEMETRY_STRIDE_FAST if movement_ema >= FAST_STRIDE_EMA_THR else TELEMETRY_STRIDE_SLOW
        if frame_idx % stride_now != 0:
            frame_idx += 1
            continue

        kpts, bbox_state, bbox_keep, lockon_count, dbg = infer_pose_fullnorm(
            interpreter, input_details, output_details,
            frame, t, movement_ema,
            bbox_state, bbox_keep,
            lockon_count
        )

        curr_conf = kpts[:, 2].astype(np.float32)
        curr_yx = kpts[:, :2].astype(np.float32)

        good = curr_conf > CONF_THRESHOLD
        score = 0.0
        has_pose_now = False

        if good.sum() >= MIN_GOOD_KPTS:
            has_pose_now = True
            if prev_pts is not None and prev_good is not None:
                both = good & prev_good
                if both.sum() >= MIN_OVERLAP_KPTS:
                    diffs = curr_yx[both] - prev_pts[both]
                    score = float(np.linalg.norm(diffs, axis=1).mean())
            prev_pts = curr_yx
            prev_good = good
        else:
            prev_pts = None
            prev_good = None
            score = 0.0
            has_pose_now = False

        # update EMA
        movement_ema = MOVEMENT_EMA_ALPHA * score + (1 - MOVEMENT_EMA_ALPHA) * movement_ema
        last_score = score
        last_has_pose = has_pose_now

        # bone ref update on stable frames (not fast, good core)
        if has_pose_now and movement_ema < 0.045 and core_mean_conf(kpts) >= 0.22:
            bone_ref = update_bone_ref(bone_ref, curr_yx, curr_conf)

        # stabilize bones when distal confidence drops
        curr_yx = stabilize_bones(curr_yx, curr_conf, bone_ref, strength=BONE_STAB_STRENGTH)

        # dynamic clamp (fast movement)
        max_jump = MAX_JOINT_JUMP_FAST if movement_ema >= FAST_MOVE_MOVEMENT_EMA_THR else MAX_JOINT_JUMP

        # smooth pose (confidence-aware + clamp)
        if has_pose_now:
            smoothed_yx, smoothed_conf = smooth_pose(
                smoothed_yx, smoothed_conf,
                curr_yx, curr_conf,
                POSE_SMOOTH_ALPHA, max_jump, FREEZE_CONF_THR
            )
        else:
            smoothed_yx, smoothed_conf = None, None

        # pause logic
        if has_pose_now:
            is_moving = movement_ema >= MOVEMENT_THRESHOLD
            if not is_moving:
                if pause_started_t is None:
                    pause_started_t = t
                    confirmed_pause = False
                if (t - pause_started_t) >= MIN_PAUSE_SECONDS and not confirmed_pause:
                    confirmed_pause = True
                    micro_pause_count += 1
            else:
                if pause_started_t is not None and confirmed_pause:
                    micro_pause_time += (t - pause_started_t)
                pause_started_t = None
                confirmed_pause = False

        history.append(float(movement_ema))
        if len(history) > window_len:
            history = history[-window_len:]

        # draw frame
        vis = frame.copy()

        if draw_crop_box and bbox_state is not None and USE_CROP_TRACKING and lockon_count >= LOCKON_GOOD_FRAMES:
            x1, y1, x2, y2 = bbox_state
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)

        if smoothed_yx is not None and smoothed_conf is not None:
            kpts_draw = np.zeros((17, 3), dtype=np.float32)
            kpts_draw[:, :2] = smoothed_yx
            kpts_draw[:, 2] = smoothed_conf

            thr = np.maximum(DRAW_THR_BY_JOINT, float(DRAW_CONF_THRESHOLD))
            if (kpts_draw[:, 2] >= thr).sum() >= 7:
                draw_pose(vis, kpts_draw, conf_thr=DRAW_CONF_THRESHOLD)

        # state label
        if not last_has_pose:
            state = "NO POSE"
            state_color = (0, 0, 255)
        else:
            if pause_started_t is None:
                state = "MOVING" if movement_ema >= MOVEMENT_THRESHOLD else "PAUSING"
                state_color = (0, 255, 0) if state == "MOVING" else (0, 165, 255)
            else:
                state = "PAUSE" if confirmed_pause else "PAUSING"
                state_color = (255, 0, 255) if confirmed_pause else (0, 165, 255)

        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        lines = [
            (f"time: {t:6.2f}s", (240, 240, 240)),
            (f"state: {state}", state_color),
            (f"move EMA: {movement_ema:.4f}", (240, 240, 240)),
            (f"move raw: {last_score:.4f}", (160, 160, 160)),
            (f"stride: {stride_now}", (200, 200, 200)),
            (f"crop: {'yes' if dbg.get('used_crop', False) else 'no'}", (200, 200, 200)),
            (f"pad: {dbg.get('pad', 0.0):.2f}", (200, 200, 200)),
            (f"lock: {dbg.get('lockon', 0)}/{LOCKON_GOOD_FRAMES}", (200, 200, 200)),
            (f"micro-pauses: {micro_pause_count}", (240, 240, 240)),
            (f"micro-pause time: {micro_pause_time:5.1f}s", (240, 240, 240)),
            (f"model: {'thunder' if USE_THUNDER else 'lightning'}", (240, 240, 240)),
        ]
        draw_panel(panel, lines, header="Climb Telemetry")
        draw_sparkline(panel, history, vmin=0.0, vmax=0.12, thresh=MOVEMENT_THRESHOLD)

        out_frame = np.hstack([vis, panel])
        writer.write(out_frame)

        frame_idx += 1

    # finalize pause time if ended mid-pause
    if pause_started_t is not None and confirmed_pause:
        end_t = frame_idx / fps_in
        micro_pause_time += (end_t - pause_started_t)

    writer.release()
    cap.release()
    return out_path, fps_out


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a climbing clip using MoveNet and output CSV/PNG + telemetry video (jump improved)."
    )
    parser.add_argument("video", help="Path to video file (mp4/mov)")
    parser.add_argument("--out", default="outputs", help="Output folder (default: ./outputs)")
    parser.add_argument("--debug", action="store_true", help="Print keypoint confidence stats on first processed frame")
    parser.add_argument("--telemetry", action="store_true", help="Write telemetry (raw AVI + optional H.264 MP4)")
    parser.add_argument("--telemetry-h264", action="store_true", help="Also write QuickTime-friendly H.264 MP4 (requires ffmpeg)")
    parser.add_argument("--no-proxy", action="store_true", help="Disable auto proxy conversion")
    parser.add_argument("--proxy-width", type=int, default=PROXY_WIDTH)
    parser.add_argument("--proxy-fps", type=int, default=PROXY_FPS)
    parser.add_argument("--thunder", action="store_true", help="Use MoveNet Thunder (more accurate, slower)")
    parser.add_argument("--no-crop", action="store_true", help="Disable crop tracking")
    parser.add_argument("--draw-crop", action="store_true", help="Draw crop box on telemetry")

    args = parser.parse_args()
    DEBUG = bool(args.debug)

    out_dir = Path(args.out)

    if args.no_proxy:
        AUTO_PROXY = False
    else:
        PROXY_WIDTH = int(args.proxy_width)
        PROXY_FPS = int(args.proxy_fps)

    if args.no_crop:
        USE_CROP_TRACKING = False

    if args.thunder:
        USE_THUNDER = True

    # update MODEL_PATH after CLI flags
    MODEL_PATH = str(Path(__file__).resolve().parent / (MODEL_THUNDER if USE_THUNDER else MODEL_LIGHTNING))

    # Use proxy for speed/stability
    video_used = make_or_use_proxy(args.video, out_dir)
    if video_used != args.video:
        print(f"[proxy] Using: {video_used}")

    interpreter, input_details, output_details = load_interpreter(MODEL_PATH)

    summary, df, pause_segments, fps_in = analyze_video(video_used, interpreter, input_details, output_details)
    print(summary)

    csv_path, png_path = save_outputs(video_used, df, pause_segments, out_dir)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")

    if args.telemetry:
        stem = Path(video_used).stem
        telemetry_raw = out_dir / f"{stem}_telemetry_v2.avi"
        out_path, fps_out = write_annotated_video(
            video_used,
            str(telemetry_raw),
            interpreter, input_details, output_details,
            draw_crop_box=bool(args.draw_crop)
        )
        print(f"Saved telemetry (raw AVI): {telemetry_raw} (fps_out≈{fps_out:.2f})")

        if args.telemetry_h264:
            telemetry_h264 = out_dir / f"{stem}_telemetry_v2_h264.mp4"
            # approximate output fps for re-encode
            fps_out_int = max(1, int(round(fps_out)))
            ffmpeg_reencode_to_h264_qt(str(telemetry_raw), str(telemetry_h264), fps=fps_out_int, max_h=1920)
            print(f"Saved telemetry (QuickTime H.264): {telemetry_h264}")