#(venv) alex@Alexs-MacBook-Pro climb_analytics % python3 analyze_yolov8_climb_tracking_sanity_v6.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264

#(venv) alex@Alexs-MacBook-Pro climb_analytics %  python3 analyze_yolov8_climb_tracking_sanity_v6.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264 
#more accuracy for jumps

# analyze_yolov8_climb_tracking_sanity_v6.py
# v6 = v5 + (C) tracker-assisted pose + (D) skeletal sanity gating + (E) torso-first robust movement + (F) richer plots/metrics
#
# Goal: make decay modeling possible by turning pose failures into "missing data" instead of spikes.
#
# Run:
#   python3 analyze_yolov8_climb_tracking_sanity_v6.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264
#
# More accuracy (slower):
#   python3 analyze_yolov8_climb_tracking_sanity_v6.py climbing_clip_1.mp4 --out outputs --model yolov8s-pose.pt --telemetry --telemetry-h264

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import subprocess
import shutil
import time

# -----------------------------
# User-facing knobs
# -----------------------------
AUTO_PROXY = True
PROXY_WIDTH = 1280
PROXY_FPS = 30

# Auto-crop (kept, but no longer relied upon as "main noise fix")
AUTO_CROP = True
CROP_STRIDE = 2
CROP_SMOOTH_ALPHA = 0.18
CROP_PAD_FRAC = 1.45        # extra conservative
CROP_MIN_SIZE = 560
CROP_OUT_HEIGHT = 900

# YOLO model
YOLO_MODEL_NAME = "yolov8n-pose.pt"  # try yolov8s-pose.pt for higher quality

# Analysis stride
ANALYSIS_STRIDE = 2
TELEMETRY_STRIDE_SLOW = 4
TELEMETRY_STRIDE_FAST = 1

# Pose confidence thresholds
KPT_CONF_THR = 0.10
MIN_GOOD_KPTS = 6
MIN_TORSO_KPTS = 4

# Movement and pauses
MOVEMENT_THRESHOLD = 0.030
MIN_PAUSE_SECONDS = 1.25
MERGE_GAP_SECONDS = 0.35

# Telemetry stride trigger
FAST_STRIDE_EMA_THR = 0.050
MOVEMENT_EMA_ALPHA = 0.25

# Active climb segmentation
ACTIVE_START_HITS = 6
ACTIVE_END_MISSES = 10
ACTIVE_MIN_BBOX_AREA_FRAC = 0.015
ACTIVE_MAX_ROOT_Y = 0.88
ACTIVE_MIN_HANDS_ABOVE_HIPS_FRAC = 0.25
ACTIVE_HANDS_WINDOW_S = 1.0

# Window features for decay
WINDOW_SECONDS = 30.0
WINDOW_STEP_SECONDS = 10.0
MIN_VALID_FRAC_FOR_DECAY = 0.60

# Sanity gating (this is the big noise killer)
BONE_JUMP_FRAC = 0.45         # reject if bone length changes >45% between frames
JOINT_TELEPORT = 0.18         # reject if wrist/ankle moves >0.18 normalized in one step
ROOT_SPEED_TELEPORT = 2.5     # reject root speed above this (norm/s)

# OneEuro smoothing (still useful, but secondary now)
ONEEURO_MIN_CUTOFF = 1.2
ONEEURO_BETA = 0.02
ONEEURO_D_CUTOFF = 1.0

# Extra metrics
ROLLING_SECONDS = 2.0

# Drawing
DRAW_CONF_THRESHOLD = 0.12
DRAW_THR_BY_JOINT = np.array([
    0.20, 0.20, 0.20, 0.20, 0.20,   # face
    0.15, 0.15,                     # shoulders
    0.20, 0.20,                     # elbows
    0.30, 0.30,                     # wrists
    0.12, 0.12,                     # hips
    0.18, 0.18,                     # knees
    0.30, 0.30                      # ankles
], dtype=np.float32)

SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

DEBUG = False

# COCO-17 indices
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 6
IDX_L_ELBOW    = 7
IDX_R_ELBOW    = 8
IDX_L_WRIST    = 9
IDX_R_WRIST    = 10
IDX_L_HIP      = 11
IDX_R_HIP      = 12
IDX_L_KNEE     = 13
IDX_R_KNEE     = 14
IDX_L_ANKLE    = 15
IDX_R_ANKLE    = 16

TORSO_IDXS = np.array([IDX_L_SHOULDER, IDX_R_SHOULDER, IDX_L_ELBOW, IDX_R_ELBOW, IDX_L_HIP, IDX_R_HIP], dtype=int)


# -----------------------
# One Euro filter
# -----------------------
def _alpha(cutoff, freq):
    te = 1.0 / max(1e-6, freq)
    tau = 1.0 / (2.0 * np.pi * max(1e-6, cutoff))
    return 1.0 / (1.0 + tau / te)

class OneEuro:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = float(t)
            return float(x)

        dt = float(t - self.t_prev)
        if dt <= 1e-6:
            dt = 1.0 / max(1e-6, self.freq)
        freq = 1.0 / dt
        self.freq = 0.9 * self.freq + 0.1 * freq

        dx = (float(x) - self.x_prev) * self.freq
        a_d = _alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, self.freq)
        x_hat = a * float(x) + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = float(t)
        return float(x_hat)

class KptSmoother:
    def __init__(self, n_kpts, freq, conf_thr):
        self.n = int(n_kpts)
        self.conf_thr = float(conf_thr)
        self.fy = [OneEuro(freq, ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF) for _ in range(self.n)]
        self.fx = [OneEuro(freq, ONEEURO_MIN_CUTOFF, ONEEURO_BETA, ONEEURO_D_CUTOFF) for _ in range(self.n)]
        self.has = [False] * self.n
        self.last_y = np.full(self.n, np.nan, dtype=np.float32)
        self.last_x = np.full(self.n, np.nan, dtype=np.float32)

    def reset(self):
        for f in self.fy: f.reset()
        for f in self.fx: f.reset()
        self.has = [False] * self.n
        self.last_y[:] = np.nan
        self.last_x[:] = np.nan

    def apply(self, kpts_yx_conf, t):
        if kpts_yx_conf is None:
            return None
        out = kpts_yx_conf.copy()
        for i in range(self.n):
            conf = float(out[i, 2])
            y = float(out[i, 0])
            x = float(out[i, 1])
            if conf >= self.conf_thr and np.isfinite(y) and np.isfinite(x):
                y_f = self.fy[i](y, t)
                x_f = self.fx[i](x, t)
                out[i, 0] = y_f
                out[i, 1] = x_f
                self.last_y[i] = y_f
                self.last_x[i] = x_f
                self.has[i] = True
            else:
                if self.has[i]:
                    out[i, 0] = float(self.last_y[i])
                    out[i, 1] = float(self.last_x[i])
        return out


# -----------------------
# ffmpeg helpers
# -----------------------
def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def ffmpeg_make_proxy(in_path: str, out_path: str, width=1280, fps=30):
    if not _have("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")
    vf = f"scale={int(width)}:-2,fps={int(fps)},format=yuv420p"
    cmd = ["ffmpeg", "-y", "-i", str(in_path), "-vf", vf,
           "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-movflags", "+faststart", "-an", str(out_path)]
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
    cmd = ["ffmpeg", "-y", "-i", str(in_path),
           "-vf", vf, "-vsync", "cfr",
           "-c:v", "libx264", "-profile:v", "high", "-level", "4.1",
           "-crf", "20", "-preset", "veryfast",
           "-movflags", "+faststart", "-video_track_timescale", "600",
           "-an", str(out_path)]
    subprocess.run(cmd, check=True)


# -----------------------
# Video helpers
# -----------------------
def open_video_or_raise(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap

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
# YOLOv8 Pose backend (with tracking)
# -----------------------
def pick_device() -> str:
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def load_yolo_pose(model_name: str):
    from ultralytics import YOLO
    return YOLO(model_name)

def run_yolo_pose(model, frame_bgr, device: str, use_track: bool = True):
    """
    Returns (kpts_yx_conf, bbox_xyxy, bbox_area_frac, has_pose)
    kpts normalized: [y,x,conf]
    """
    H, W = frame_bgr.shape[:2]

    # Try tracking for more stable identity/bbox
    try:
        if use_track:
            results = model.track(
                source=frame_bgr,
                device=device,
                verbose=False,
                conf=0.25,
                iou=0.5,
                imgsz=640,
                persist=True,
                tracker="bytetrack.yaml",
            )
        else:
            results = model.predict(
                source=frame_bgr,
                device=device,
                verbose=False,
                conf=0.25,
                iou=0.5,
                imgsz=640,
            )
    except Exception:
        # if tracker isn't available, fallback to predict
        results = model.predict(
            source=frame_bgr,
            device=device,
            verbose=False,
            conf=0.25,
            iou=0.5,
            imgsz=640,
        )

    if not results:
        return None, None, 0.0, False

    r = results[0]
    if r.boxes is None or r.keypoints is None:
        return None, None, 0.0, False

    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
    kpts_xyn = r.keypoints.xyn
    kpts_conf = r.keypoints.conf
    if kpts_xyn is None or kpts_conf is None:
        return None, None, 0.0, False

    kpts_xyn = kpts_xyn.cpu().numpy() if hasattr(kpts_xyn, "cpu") else np.array(kpts_xyn)
    kpts_conf = kpts_conf.cpu().numpy() if hasattr(kpts_conf, "cpu") else np.array(kpts_conf)

    if boxes.shape[0] == 0 or kpts_xyn.shape[0] == 0:
        return None, None, 0.0, False

    # pick largest bbox (still good when only one climber)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    bb = boxes[idx].astype(int).tolist()
    x1, y1, x2, y2 = bb
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    bb = (x1, y1, x2, y2)
    area_frac = float(max(1, (x2 - x1)) * max(1, (y2 - y1))) / float(W * H)

    xy = kpts_xyn[idx]  # [x,y]
    cf = kpts_conf[idx]
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = xy[:, 1]  # y
    kpts[:, 1] = xy[:, 0]  # x
    kpts[:, 2] = cf
    return kpts, bb, area_frac, True


# -----------------------
# Auto-crop (bbox track -> cropped MP4)
# -----------------------
def _clamp_bb(bb, W, H):
    x1, y1, x2, y2 = bb
    x1 = max(0, min(W - 2, x1))
    y1 = max(0, min(H - 2, y1))
    x2 = max(x1 + 2, min(W, x2))
    y2 = max(y1 + 2, min(H, y2))
    return (x1, y1, x2, y2)

def _expand_bbox_safe(bb, W, H, pad_frac=1.10, min_size=520, aspect=9/16):
    x1, y1, x2, y2 = bb
    bw = max(2, x2 - x1)
    bh = max(2, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw2 = max(min_size, int(round(bw * (1.0 + pad_frac * 0.85))))
    bh2 = max(min_size, int(round(bh * (1.0 + pad_frac * 1.25))))

    target_bh = int(round(bw2 / max(1e-6, aspect)))
    if target_bh > bh2:
        bh2 = target_bh
    else:
        target_bw = int(round(bh2 * aspect))
        if target_bw > bw2:
            bw2 = target_bw

    nx1 = int(round(cx - bw2 / 2))
    ny1 = int(round(cy - bh2 / 2))
    nx2 = nx1 + bw2
    ny2 = ny1 + bh2
    return _clamp_bb((nx1, ny1, nx2, ny2), W, H)

def _smooth_bb_hysteresis(prev, curr, alpha=0.18, shrink_alpha=0.05):
    if prev is None:
        return curr
    px1, py1, px2, py2 = prev
    x1, y1, x2, y2 = curr
    nx1 = alpha * x1 + (1 - alpha) * px1
    ny1 = alpha * y1 + (1 - alpha) * py1
    nx2 = alpha * x2 + (1 - alpha) * px2
    ny2 = alpha * y2 + (1 - alpha) * py2

    if (x2 - x1) < (px2 - px1):
        nx1 = shrink_alpha * x1 + (1 - shrink_alpha) * px1
        nx2 = shrink_alpha * x2 + (1 - shrink_alpha) * px2
    if (y2 - y1) < (py2 - py1):
        ny1 = shrink_alpha * y1 + (1 - shrink_alpha) * py1
        ny2 = shrink_alpha * y2 + (1 - shrink_alpha) * py2

    return (int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2)))

def build_and_save_cropped_video(video_path: str, out_dir: Path, model, device: str):
    p = Path(video_path)
    crop_path = out_dir / f"{p.stem}_crop_safe_v6.mp4"
    if crop_path.exists():
        return str(crop_path)

    cap = open_video_or_raise(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1:
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    sampled = {}
    prev_smooth = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % CROP_STRIDE == 0:
            kpts, bb, _, ok_pose = run_yolo_pose(model, frame, device=device, use_track=True)
            if ok_pose and bb is not None:
                safe = _expand_bbox_safe(bb, W, H, pad_frac=CROP_PAD_FRAC, min_size=CROP_MIN_SIZE, aspect=9/16)
                safe = _smooth_bb_hysteresis(prev_smooth, safe, alpha=CROP_SMOOTH_ALPHA, shrink_alpha=0.05)
                prev_smooth = safe
                sampled[frame_idx] = safe
        frame_idx += 1
    cap.release()

    if not sampled:
        print("[crop] No detections; using original.")
        return video_path

    total = n_frames if n_frames > 0 else (max(sampled.keys()) + 1)
    bboxes = [None] * total
    keys = sorted(sampled.keys())

    first_k = keys[0]
    for i in range(0, min(first_k + 1, total)):
        bboxes[i] = sampled[first_k]

    for a, b in zip(keys[:-1], keys[1:]):
        bba = sampled[a]
        bbb = sampled[b]
        span = max(1, b - a)
        for i in range(a, min(b + 1, total)):
            t = (i - a) / span
            interp = (
                int(round((1 - t) * bba[0] + t * bbb[0])),
                int(round((1 - t) * bba[1] + t * bbb[1])),
                int(round((1 - t) * bba[2] + t * bbb[2])),
                int(round((1 - t) * bba[3] + t * bbb[3])),
            )
            bboxes[i] = _clamp_bb(interp, W, H)

    last_k = keys[-1]
    for i in range(last_k, total):
        bboxes[i] = sampled[last_k]

    out_h = int(CROP_OUT_HEIGHT)
    out_w = int(round(out_h * (9/16)))
    out_w = max(360, min(1920, out_w))

    cap = open_video_or_raise(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(crop_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"[crop] Could not open VideoWriter: {crop_path}")

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bb = bboxes[min(fi, len(bboxes) - 1)]
        x1, y1, x2, y2 = bb
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        crop_rs = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(crop_rs)
        fi += 1
    writer.release()
    cap.release()

    if _have("ffmpeg"):
        qt_path = out_dir / f"{p.stem}_crop_safe_v6_h264.mp4"
        try:
            ffmpeg_reencode_to_h264_qt(str(crop_path), str(qt_path), fps=int(round(fps)), max_h=out_h)
            return str(qt_path)
        except Exception:
            return str(crop_path)

    return str(crop_path)


# -----------------------
# Movement + segmentation + sanity gating
# -----------------------
def mid_hip_root(kpts):
    y = 0.5 * (float(kpts[IDX_L_HIP, 0]) + float(kpts[IDX_R_HIP, 0]))
    x = 0.5 * (float(kpts[IDX_L_HIP, 1]) + float(kpts[IDX_R_HIP, 1]))
    return np.array([y, x], dtype=np.float32)

def mid_shoulder(kpts):
    y = 0.5 * (float(kpts[IDX_L_SHOULDER, 0]) + float(kpts[IDX_R_SHOULDER, 0]))
    x = 0.5 * (float(kpts[IDX_L_SHOULDER, 1]) + float(kpts[IDX_R_SHOULDER, 1]))
    return np.array([y, x], dtype=np.float32)

def hands_above_hips(kpts):
    lw_y = float(kpts[IDX_L_WRIST, 0]);  rw_y = float(kpts[IDX_R_WRIST, 0])
    lh_y = float(kpts[IDX_L_HIP, 0]);    rh_y = float(kpts[IDX_R_HIP, 0])
    hips_y = 0.5 * (lh_y + rh_y)
    return (lw_y < hips_y) or (rw_y < hips_y)

def _bone_len(kpts, a, b, thr=KPT_CONF_THR):
    if kpts is None:
        return np.nan
    if float(kpts[a, 2]) < thr or float(kpts[b, 2]) < thr:
        return np.nan
    dy = float(kpts[a, 0] - kpts[b, 0])
    dx = float(kpts[a, 1] - kpts[b, 1])
    return float(np.sqrt(dx*dx + dy*dy))

def pose_sanity_ok(prev, curr, thr=KPT_CONF_THR):
    """
    Reject physically impossible frame-to-frame jumps.
    """
    if prev is None or curr is None:
        return True, "no_prev"

    bones = [
        (IDX_L_SHOULDER, IDX_L_ELBOW),
        (IDX_L_ELBOW, IDX_L_WRIST),
        (IDX_R_SHOULDER, IDX_R_ELBOW),
        (IDX_R_ELBOW, IDX_R_WRIST),
        (IDX_L_HIP, IDX_L_KNEE),
        (IDX_L_KNEE, IDX_L_ANKLE),
        (IDX_R_HIP, IDX_R_KNEE),
        (IDX_R_KNEE, IDX_R_ANKLE),
    ]
    for a, b in bones:
        l0 = _bone_len(prev, a, b, thr)
        l1 = _bone_len(curr, a, b, thr)
        if np.isfinite(l0) and np.isfinite(l1):
            if abs(l1 - l0) / max(1e-6, l0) > BONE_JUMP_FRAC:
                return False, "bone_jump"

    fast_joints = [IDX_L_WRIST, IDX_R_WRIST, IDX_L_ANKLE, IDX_R_ANKLE]
    for j in fast_joints:
        if float(prev[j, 2]) >= thr and float(curr[j, 2]) >= thr:
            dy = float(curr[j, 0] - prev[j, 0])
            dx = float(curr[j, 1] - prev[j, 1])
            d = float(np.sqrt(dx*dx + dy*dy))
            if d > JOINT_TELEPORT:
                return False, "teleport"

    return True, "ok"

def movement_masked(prev, curr, mask, conf_thr=KPT_CONF_THR):
    if prev is None or curr is None:
        return np.nan, 0, 0.0
    prev_conf = prev[:, 2]
    curr_conf = curr[:, 2]
    good = mask & (prev_conf >= conf_thr) & (curr_conf >= conf_thr)
    n = int(good.sum())
    if n < 3:
        q = float(np.nanmean(curr_conf[mask])) if mask.any() else float(np.nanmean(curr_conf))
        return np.nan, n, q

    pr = mid_hip_root(prev)
    cr = mid_hip_root(curr)
    prev_rel = prev[:, :2] - pr[None, :]
    curr_rel = curr[:, :2] - cr[None, :]

    diffs = curr_rel[good] - prev_rel[good]
    per = np.linalg.norm(diffs, axis=1)
    score = float(np.median(per))
    q = float(curr_conf[good].mean())
    return score, n, q

def movement_body_relative_torso_first(prev, curr, conf_thr=KPT_CONF_THR):
    """
    Torso score always computed; only use full body if plenty of joints exist.
    """
    torso_mask = np.zeros(17, dtype=bool)
    torso_mask[TORSO_IDXS] = True
    score_t, n_t, q_t = movement_masked(prev, curr, torso_mask, conf_thr=conf_thr)

    full_mask = np.ones(17, dtype=bool)
    score_f, n_f, q_f = movement_masked(prev, curr, full_mask, conf_thr=conf_thr)

    if np.isfinite(score_f) and n_f >= MIN_GOOD_KPTS:
        return score_f, n_f, q_f, "full"
    return score_t, n_t, q_t, "torso"

def _segments_from_boolean(times, is_true):
    segs = []
    in_seg = False
    start = None
    for t, p in zip(times, is_true):
        if p and not in_seg:
            in_seg = True
            start = t
        elif (not p) and in_seg:
            in_seg = False
            segs.append((float(start), float(t)))
    if in_seg and start is not None and len(times):
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

def compute_active_climb_mask(times, has_pose, root_y, bbox_area_frac, hands_flag, fps_eff):
    n = len(times)
    active = np.zeros(n, dtype=bool)

    win = max(1, int(round(ACTIVE_HANDS_WINDOW_S * fps_eff)))
    hands_frac = np.zeros(n, dtype=np.float32)
    s = 0
    for i in range(n):
        s += 1 if hands_flag[i] else 0
        if i >= win:
            s -= 1 if hands_flag[i - win] else 0
        denom = min(i + 1, win)
        hands_frac[i] = float(s) / float(max(1, denom))

    on = False
    hits = 0
    misses = 0

    for i in range(n):
        cand = (
            bool(has_pose[i]) and
            float(bbox_area_frac[i]) >= ACTIVE_MIN_BBOX_AREA_FRAC and
            float(root_y[i]) <= ACTIVE_MAX_ROOT_Y and
            float(hands_frac[i]) >= ACTIVE_MIN_HANDS_ABOVE_HIPS_FRAC
        )

        if not on:
            if cand:
                hits += 1
                if hits >= ACTIVE_START_HITS:
                    on = True
                    misses = 0
            else:
                hits = 0
        else:
            if cand:
                misses = 0
            else:
                misses += 1
                if misses >= ACTIVE_END_MISSES:
                    on = False
                    hits = 0

        active[i] = on

    return active


# -----------------------
# Drawing
# -----------------------
def _to_pixel_xy(kpts_yx, w, h):
    ys = (kpts_yx[:, 0] * h).astype(int)
    xs = (kpts_yx[:, 1] * w).astype(int)
    return xs, ys

def draw_pose(frame_bgr, kpts, conf_thr=DRAW_CONF_THRESHOLD):
    h, w = frame_bgr.shape[:2]
    conf = kpts[:, 2].astype(np.float32)
    pts_yx = kpts[:, :2].astype(np.float32)
    thr = np.maximum(DRAW_THR_BY_JOINT, float(conf_thr))
    good = conf >= thr
    xs, ys = _to_pixel_xy(pts_yx, w, h)

    for a, b in SKELETON_EDGES:
        if good[a] and good[b]:
            cv2.line(frame_bgr, (xs[a], ys[a]), (xs[b], ys[b]), (0, 255, 0), 2, cv2.LINE_AA)
    for i in range(17):
        if good[i]:
            cv2.circle(frame_bgr, (xs[i], ys[i]), 3, (0, 255, 255), -1, cv2.LINE_AA)

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
    vals = np.array(values, dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0)
    vals = np.clip(vals, vmin, vmax)
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
# Metrics helpers
# -----------------------
def _safe_joint_y(kpts, idx, conf_thr=KPT_CONF_THR):
    if kpts is None:
        return np.nan
    return float(kpts[idx, 0]) if float(kpts[idx, 2]) >= conf_thr else np.nan

def _safe_dist(kpts, a, b, conf_thr=KPT_CONF_THR):
    if kpts is None:
        return np.nan
    if float(kpts[a, 2]) < conf_thr or float(kpts[b, 2]) < conf_thr:
        return np.nan
    dy = float(kpts[a, 0] - kpts[b, 0])
    dx = float(kpts[a, 1] - kpts[b, 1])
    return float(np.sqrt(dy * dy + dx * dx))

def _rolling_std(values: np.ndarray, win: int):
    out = np.full_like(values, np.nan, dtype=np.float32)
    n = len(values)
    for i in range(n):
        s = max(0, i - win + 1)
        seg = values[s:i+1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= max(3, win // 3):
            out[i] = float(np.std(seg))
    return out

def _nan_diff(values: np.ndarray):
    out = np.full_like(values, np.nan, dtype=np.float32)
    for i in range(1, len(values)):
        a = values[i - 1]
        b = values[i]
        if np.isfinite(a) and np.isfinite(b):
            out[i] = float(b - a)
    return out

def _zscore(arr: np.ndarray):
    mu = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
    sd = float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan
    if not np.isfinite(mu) or not np.isfinite(sd) or sd < 1e-9:
        return np.full_like(arr, np.nan, dtype=np.float32)
    return (arr - mu) / sd


# -----------------------
# Decay windows
# -----------------------
def compute_decay_windows(df_active, window_s, step_s):
    if len(df_active) == 0:
        return pd.DataFrame()

    t = df_active["t"].values.astype(float)
    t0 = float(np.nanmin(t))
    t1 = float(np.nanmax(t))

    rows = []
    start = t0
    while start + window_s <= t1 + 1e-6:
        end = start + window_s
        win = df_active[(df_active["t"] >= start) & (df_active["t"] < end)]
        if len(win) >= 3:
            mv = win["move_body"].values.astype(float)
            valid = np.isfinite(mv)
            valid_frac = float(valid.mean()) if len(mv) else 0.0

            mean_move = float(np.nanmean(mv)) if valid.any() else np.nan
            mean_ema = float(np.nanmean(win["move_ema"].values.astype(float)))
            mean_jerk = float(np.nanmean(win["move_jerk"].values.astype(float)))
            mean_var = float(np.nanmean(win["move_ema_roll_std"].values.astype(float)))
            mean_q = float(np.nanmean(win["pose_quality"].values.astype(float)))
            mean_speed = float(np.nanmean(win["root_speed"].values.astype(float)))

            pause_frac = float((win.loc[valid, "is_paused"] == True).mean()) if valid.any() else np.nan

            segs = _segments_from_boolean(win["t"].values[valid], win["is_paused"].values[valid])
            segs = _filter_and_merge_segments(segs, MIN_PAUSE_SECONDS, MERGE_GAP_SECONDS)
            mp_count = int(len(segs))
            mp_time = float(sum(e - s for s, e in segs))

            rows.append({
                "t_start": float(start),
                "t_end": float(end),
                "valid_frame_fraction": valid_frac,
                "mean_move_body": mean_move,
                "mean_move_ema": mean_ema,
                "mean_move_jerk": mean_jerk,
                "mean_move_variability": mean_var,
                "pause_fraction_valid": pause_frac,
                "micro_pause_count": mp_count,
                "micro_pause_time_s": mp_time,
                "mean_pose_quality": mean_q,
                "mean_root_speed": mean_speed,
            })

        start += step_s

    out = pd.DataFrame(rows)
    if len(out):
        z_jerk = _zscore(out["mean_move_jerk"].values.astype(np.float32))
        z_var  = _zscore(out["mean_move_variability"].values.astype(np.float32))
        z_pause = _zscore(out["pause_fraction_valid"].values.astype(np.float32))
        slop = z_jerk + z_var + 0.75 * z_pause
        out["sloppiness_index"] = slop
        out["stability_index"] = -slop

        bad = out["valid_frame_fraction"].astype(float) < MIN_VALID_FRAC_FOR_DECAY
        out.loc[bad, ["sloppiness_index", "stability_index"]] = np.nan

    return out


# -----------------------
# Main analysis
# -----------------------
def analyze(video_path: str, model, device: str, t_start: float | None, t_end: float | None):
    cap = open_video_or_raise(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1:
        fps = 30.0

    fps_eff = fps / max(1, ANALYSIS_STRIDE)
    smoother = KptSmoother(n_kpts=17, freq=fps_eff, conf_thr=KPT_CONF_THR)

    times = []
    has_pose = []
    valid_pose = []
    sanity_reason = []
    bbox_area = []
    root_y = []
    hands_flag = []

    move_body = []
    move_kind = []
    move_ema = []
    n_used = []
    pose_quality = []

    hip_y = []
    shoulder_y = []
    hand_y_min = []
    foot_y_min = []
    reach_span = []
    stance_width = []
    root_speed = []

    prev_root = None
    ema = 0.0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        if t_start is not None and t < t_start:
            frame_idx += 1
            continue
        if t_end is not None and t > t_end:
            break

        if frame_idx % ANALYSIS_STRIDE != 0:
            frame_idx += 1
            continue

        kpts, bb, area_frac, ok_pose = run_yolo_pose(model, frame, device=device, use_track=True)

        if not ok_pose or kpts is None:
            times.append(float(t))
            has_pose.append(False)
            valid_pose.append(False)
            sanity_reason.append("no_pose")
            bbox_area.append(0.0)
            root_y.append(1.0)
            hands_flag.append(False)

            move_body.append(np.nan)
            move_kind.append("none")
            ema = (1 - MOVEMENT_EMA_ALPHA) * ema
            move_ema.append(float(ema))
            n_used.append(0)
            pose_quality.append(0.0)

            hip_y.append(np.nan)
            shoulder_y.append(np.nan)
            hand_y_min.append(np.nan)
            foot_y_min.append(np.nan)
            reach_span.append(np.nan)
            stance_width.append(np.nan)
            root_speed.append(np.nan)

            prev_kpts = None
            prev_root = None
            smoother.reset()
            frame_idx += 1
            continue

        # Smooth keypoints first
        kpts_s = smoother.apply(kpts, t)

        # sanity gating against prev accepted
        ok_sanity, why = pose_sanity_ok(prev_kpts, kpts_s, thr=KPT_CONF_THR)

        # compute movement score torso-first, but ONLY if sanity ok and prev exists
        score = np.nan
        kind = "none"
        n = int((kpts_s[:, 2] >= KPT_CONF_THR).sum())
        q = float(np.nanmean(kpts_s[:, 2]))

        if prev_kpts is not None and ok_sanity:
            score, n_use, q_use, kind = movement_body_relative_torso_first(prev_kpts, kpts_s, conf_thr=KPT_CONF_THR)
            n = int(n_use)
            q = float(q_use)

        # accept frame if sanity ok AND score is finite OR at least torso joints exist
        torso_good = int(((kpts_s[:, 2] >= KPT_CONF_THR) & np.isin(np.arange(17), TORSO_IDXS)).sum())
        is_valid = bool(ok_sanity and (np.isfinite(score) or torso_good >= MIN_TORSO_KPTS))

        # update EMA only on valid + finite score
        if is_valid and np.isfinite(score):
            ema = MOVEMENT_EMA_ALPHA * float(score) + (1 - MOVEMENT_EMA_ALPHA) * ema
            prev_kpts = kpts_s
        else:
            # do not update prev_kpts => prevents bad frames from contaminating next diff
            ema = (1 - MOVEMENT_EMA_ALPHA) * ema

        # kinematic features (only if pose exists; validity affects some)
        r = mid_hip_root(kpts_s)
        hy = hands_above_hips(kpts_s)

        dt = float(max(1e-6, ANALYSIS_STRIDE / fps))
        spd = np.nan
        if prev_root is not None:
            dy = float(r[0] - prev_root[0])
            dx = float(r[1] - prev_root[1])
            dist = float(np.sqrt(dx*dx + dy*dy))
            spd = dist / dt
            if spd > ROOT_SPEED_TELEPORT:
                spd = np.nan
        prev_root = r.copy()

        sh = mid_shoulder(kpts_s)

        lw = _safe_joint_y(kpts_s, IDX_L_WRIST)
        rw = _safe_joint_y(kpts_s, IDX_R_WRIST)
        handmin = np.nanmin([lw, rw]) if (np.isfinite(lw) or np.isfinite(rw)) else np.nan

        la = _safe_joint_y(kpts_s, IDX_L_ANKLE)
        ra = _safe_joint_y(kpts_s, IDX_R_ANKLE)
        footmin = np.nanmin([la, ra]) if (np.isfinite(la) or np.isfinite(ra)) else np.nan

        reach = _safe_dist(kpts_s, IDX_L_WRIST, IDX_R_WRIST)
        stance = _safe_dist(kpts_s, IDX_L_ANKLE, IDX_R_ANKLE)

        times.append(float(t))
        has_pose.append(True)
        valid_pose.append(bool(is_valid and np.isfinite(score)))
        sanity_reason.append(why if ok_sanity else why)
        bbox_area.append(float(area_frac))
        root_y.append(float(r[0]))
        hands_flag.append(bool(hy))

        move_body.append(float(score) if (is_valid and np.isfinite(score)) else np.nan)
        move_kind.append(kind if is_valid else "rejected")
        move_ema.append(float(ema))
        n_used.append(int(n))
        pose_quality.append(float(q))

        hip_y.append(float(r[0]))
        shoulder_y.append(float(sh[0]))
        hand_y_min.append(float(handmin) if np.isfinite(handmin) else np.nan)
        foot_y_min.append(float(footmin) if np.isfinite(footmin) else np.nan)
        reach_span.append(float(reach) if np.isfinite(reach) else np.nan)
        stance_width.append(float(stance) if np.isfinite(stance) else np.nan)
        root_speed.append(float(spd) if np.isfinite(spd) else np.nan)

        frame_idx += 1

    cap.release()

    df = pd.DataFrame({
        "t": times,
        "has_pose": has_pose,
        "valid_pose": valid_pose,
        "sanity_reason": sanity_reason,
        "bbox_area_frac": bbox_area,
        "root_y": root_y,
        "hands_above_hips": hands_flag,
        "move_body": move_body,
        "move_kind": move_kind,
        "move_ema": move_ema,
        "n_used": n_used,
        "pose_quality": pose_quality,
        "hip_y": hip_y,
        "shoulder_y": shoulder_y,
        "hand_y_min": hand_y_min,
        "foot_y_min": foot_y_min,
        "reach_span": reach_span,
        "stance_width": stance_width,
        "root_speed": root_speed,
    })

    # active segmentation uses has_pose/root/bbox/hands (not movement)
    active = compute_active_climb_mask(
        df["t"].values,
        df["has_pose"].values,
        df["root_y"].values,
        df["bbox_area_frac"].values,
        df["hands_above_hips"].values,
        fps_eff=fps_eff
    )
    df["active"] = active

    # pause logic only when active AND movement valid
    df["is_moving"] = False
    df["is_paused"] = False
    act = df["active"].values.astype(bool)
    mv = df["move_body"].values.astype(float)
    valid_mv = np.isfinite(mv)
    idx = act & valid_mv
    df.loc[idx, "is_moving"] = mv[idx] >= MOVEMENT_THRESHOLD
    df.loc[idx, "is_paused"] = mv[idx] < MOVEMENT_THRESHOLD

    # jerk + variability
    ema_arr = df["move_ema"].values.astype(np.float32)
    dt = float(max(1e-6, ANALYSIS_STRIDE / fps))
    ema_diff = _nan_diff(ema_arr)
    df["move_jerk"] = np.abs(ema_diff) / dt
    roll_win = max(3, int(round(ROLLING_SECONDS * fps_eff)))
    df["move_ema_roll_std"] = _rolling_std(ema_arr, roll_win)

    df_active = df[df["active"] == True].copy()

    # micro pauses from active+valid
    mv_a = df_active["move_body"].values.astype(float)
    valid_a = np.isfinite(mv_a)
    segs = _segments_from_boolean(df_active["t"].values[valid_a], df_active["is_paused"].values[valid_a])
    pause_segments = _filter_and_merge_segments(segs, MIN_PAUSE_SECONDS, MERGE_GAP_SECONDS)

    duration = float(df["t"].iloc[-1]) if len(df) else 0.0
    active_duration = float(df_active["t"].iloc[-1] - df_active["t"].iloc[0]) if len(df_active) >= 2 else 0.0
    pause_time = float(sum(e - s for s, e in pause_segments))
    pause_count = int(len(pause_segments))
    move_time = float(max(0.0, active_duration - pause_time))
    valid_active_frac = float(np.isfinite(df_active["move_body"].values.astype(float)).mean()) if len(df_active) else 0.0

    summary = {
        "video": Path(video_path).name,
        "duration_s": round(duration, 2),
        "active_climb_duration_s": round(active_duration, 2),
        "micro_pause_time_s": round(pause_time, 2),
        "micro_pause_count": pause_count,
        "active_move_time_s": round(move_time, 2),
        "avg_move_body_active": round(float(np.nanmean(df_active["move_body"].values.astype(float))) if len(df_active) else 0.0, 6),
        "avg_move_ema_active": round(float(np.nanmean(df_active["move_ema"].values.astype(float))) if len(df_active) else 0.0, 6),
        "avg_jerk_active": round(float(np.nanmean(df_active["move_jerk"].values.astype(float))) if len(df_active) else 0.0, 6),
        "avg_variability_active": round(float(np.nanmean(df_active["move_ema_roll_std"].values.astype(float))) if len(df_active) else 0.0, 6),
        "pose_detect_rate": round(float(df["has_pose"].mean()) if len(df) else 0.0, 3),
        "valid_active_frame_fraction": round(valid_active_frac, 3),
        "analysis_fps": round(float(fps_eff), 2),
        "device": device,
        "model": YOLO_MODEL_NAME,
        "version": "v6"
    }

    decay_df = compute_decay_windows(df_active, WINDOW_SECONDS, WINDOW_STEP_SECONDS)
    return summary, df, df_active, pause_segments, decay_df, fps


# -----------------------
# Telemetry
# -----------------------
def write_telemetry(video_path: str, out_path: str, model, device: str, panel_w: int = 320,
                    t_start: float | None = None, t_end: float | None = None,
                    draw_bbox: bool = True):
    cap0 = open_video_or_raise(video_path)
    fps_in = float(cap0.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps_in <= 1:
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
    fps_out = float(fps_in / max(1, TELEMETRY_STRIDE_SLOW))

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(out_path, fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter at: {out_path}")

    active_on = False
    hits = 0
    misses = 0

    hands_win = max(1, int(round(ACTIVE_HANDS_WINDOW_S * fps_out)))
    hands_hist = []

    ema = 0.0
    prev_kpts = None
    smoother = KptSmoother(n_kpts=17, freq=fps_out, conf_thr=KPT_CONF_THR)

    micro_pause_count = 0
    micro_pause_time = 0.0
    pause_started_t = None
    confirmed_pause = False

    history = []
    window_len = max(10, int(12.0 * fps_out))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps_in
        if t_start is not None and t < t_start:
            frame_idx += 1
            continue
        if t_end is not None and t > t_end:
            break

        stride_now = TELEMETRY_STRIDE_FAST if ema >= FAST_STRIDE_EMA_THR else TELEMETRY_STRIDE_SLOW
        if frame_idx % stride_now != 0:
            frame_idx += 1
            continue

        kpts, bb, area_frac, ok_pose = run_yolo_pose(model, frame, device=device, use_track=True)
        vis = frame.copy()
        has_pose = bool(ok_pose and kpts is not None)

        move = np.nan
        rooty = 1.0
        hands = False
        q = 0.0
        n = 0
        sanity = "no_pose"
        kind = "none"

        if has_pose:
            kpts_s = smoother.apply(kpts, t)
            ok_s, why = pose_sanity_ok(prev_kpts, kpts_s, thr=KPT_CONF_THR)
            sanity = why if ok_s else why

            root = mid_hip_root(kpts_s)
            rooty = float(root[0])
            hands = hands_above_hips(kpts_s)

            if prev_kpts is not None and ok_s:
                score, n_use, q_use, kind = movement_body_relative_torso_first(prev_kpts, kpts_s, conf_thr=KPT_CONF_THR)
                if np.isfinite(score):

                    ema = MOVEMENT_EMA_ALPHA * float(score) + (1 - MOVEMENT_EMA_ALPHA) * ema
                    prev_kpts = kpts_s
                else:
                    ema = (1 - MOVEMENT_EMA_ALPHA) * ema
            else:
                prev_kpts = kpts_s if ok_s else None
                ema = (1 - MOVEMENT_EMA_ALPHA) * ema

            if draw_bbox and bb is not None:
                x1, y1, x2, y2 = bb
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)
            draw_pose(vis, kpts_s, conf_thr=DRAW_CONF_THRESHOLD)
        else:
            prev_kpts = None
            smoother.reset()
            ema = (1 - MOVEMENT_EMA_ALPHA) * ema

        hands_hist.append(1 if hands else 0)
        if len(hands_hist) > hands_win:
            hands_hist = hands_hist[-hands_win:]
        hands_frac = float(sum(hands_hist)) / float(max(1, len(hands_hist)))

        cand = (
            has_pose and
            float(area_frac) >= ACTIVE_MIN_BBOX_AREA_FRAC and
            float(rooty) <= ACTIVE_MAX_ROOT_Y and
            float(hands_frac) >= ACTIVE_MIN_HANDS_ABOVE_HIPS_FRAC
        )

        if not active_on:
            if cand:
                hits += 1
                if hits >= ACTIVE_START_HITS:
                    active_on = True
                    misses = 0
            else:
                hits = 0
        else:
            if cand:
                misses = 0
            else:
                misses += 1
                if misses >= ACTIVE_END_MISSES:
                    active_on = False
                    hits = 0
                    pause_started_t = None
                    confirmed_pause = False

        move_valid = np.isfinite(move)

        if active_on and has_pose and move_valid:
            is_moving = ema >= MOVEMENT_THRESHOLD
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
        else:
            pause_started_t = None
            confirmed_pause = False

        history.append(float(ema))
        if len(history) > window_len:
            history = history[-window_len:]

        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        if not has_pose:
            state = "NO POSE"
            color = (0, 0, 255)
        else:
            if not active_on:
                state = "INACTIVE"
                color = (200, 200, 200)
            else:
                if not move_valid:
                    state = "INVALID"
                    color = (0, 0, 255)
                elif ema >= MOVEMENT_THRESHOLD:
                    state = "MOVING"
                    color = (0, 255, 0)
                else:
                    state = "PAUSING" if not confirmed_pause else "PAUSE"
                    color = (0, 165, 255) if not confirmed_pause else (255, 0, 255)

        lines = [
            (f"time: {t:6.2f}s", (240, 240, 240)),
            (f"state: {state}", color),
            (f"move: {move:.4f}" if move_valid else "move: NaN", (240, 240, 240)),
            (f"ema: {ema:.4f}", (240, 240, 240)),
            (f"kind: {kind}", (200, 200, 200)),
            (f"n_used: {n}", (200, 200, 200)),
            (f"q: {q:.2f}", (200, 200, 200)),
            (f"sanity: {sanity}", (200, 200, 200)),
            (f"stride: {stride_now}", (200, 200, 200)),
            (f"active: {'yes' if active_on else 'no'}", (240, 240, 240)),
            (f"bbox area: {float(area_frac):.3f}", (200, 200, 200)),
            (f"root_y: {rooty:.3f}", (200, 200, 200)),
            (f"hands frac: {hands_frac:.2f}", (200, 200, 200)),
            (f"micro-pauses: {micro_pause_count}", (240, 240, 240)),
            (f"micro-pause time: {micro_pause_time:5.1f}s", (240, 240, 240)),
            (f"device: {device}", (240, 240, 240)),
            (f"model: {YOLO_MODEL_NAME}", (240, 240, 240)),
            ("v6 (track+sanity)", (200, 200, 200)),
        ]
        draw_panel(panel, lines, header="Climb Telemetry (YOLO Pose v6)")
        draw_sparkline(panel, history, vmin=0.0, vmax=0.12, thresh=MOVEMENT_THRESHOLD)

        out_frame = np.hstack([vis, panel])
        writer.write(out_frame)

        frame_idx += 1

    if pause_started_t is not None and confirmed_pause:
        end_t = frame_idx / fps_in
        micro_pause_time += (end_t - pause_started_t)

    writer.release()
    cap.release()
    return out_path, fps_out


# -----------------------
# Plotting
# -----------------------
def _shade_active(ax, df):
    active = df["active"].values.astype(bool)
    if active.any():
        segs = _segments_from_boolean(df["t"].values, active)
        for s, e in segs:
            ax.axvspan(s, e, alpha=0.15)

def save_plot(path: Path, fig):
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)

def make_plots(out_dir: Path, stem: str, df: pd.DataFrame, df_active: pd.DataFrame, decay_df: pd.DataFrame):
    # Movement: show NaNs as gaps (this is what you WANT)
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["move_body"], label="move_body (gated)")
    ax.plot(df["t"], df["move_ema"], label="move_ema", alpha=0.85)
    ax.axhline(MOVEMENT_THRESHOLD, linestyle="--", label="thr")
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Body-relative movement")
    ax.set_title(f"{stem} movement v6 (active shaded)")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_movement.png", fig)

    # Validity and pose quality
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["pose_quality"], label="pose_quality")
    ax.plot(df["t"], df["n_used"], label="n_used", alpha=0.85)
    ax.plot(df["t"], df["valid_pose"].astype(int), label="valid_pose (0/1)", alpha=0.85)
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{stem} pose quality + validity")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_pose_validity.png", fig)

    # Heights
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["hip_y"], label="hip_y")
    ax.plot(df["t"], df["shoulder_y"], label="shoulder_y", alpha=0.85)
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized y (smaller=up)")
    ax.set_title(f"{stem} hip/shoulder height")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_heights.png", fig)

    fig, ax = plt.subplots()
    ax.plot(df["t"], df["hand_y_min"], label="hand_y_min")
    ax.plot(df["t"], df["foot_y_min"], label="foot_y_min", alpha=0.85)
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized y (smaller=up)")
    ax.set_title(f"{stem} hand/foot min height")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_hand_foot.png", fig)

    # Reach/stance
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["reach_span"], label="reach_span")
    ax.plot(df["t"], df["stance_width"], label="stance_width", alpha=0.85)
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{stem} reach/stance")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_reach_stance.png", fig)

    # Speed + jerk + variability
    fig, ax = plt.subplots()
    ax.plot(df["t"], df["root_speed"], label="root_speed (norm/s)")
    ax.plot(df["t"], df["move_jerk"], label="move_jerk", alpha=0.85)
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{stem} root travel + jerk")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_speed_jerk.png", fig)

    fig, ax = plt.subplots()
    ax.plot(df["t"], df["move_ema_roll_std"], label="move_ema rolling std")
    _shade_active(ax, df)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{stem} movement variability (rolling)")
    ax.legend()
    save_plot(out_dir / f"{stem}_plot_variability.png", fig)

    # Active distribution
    if len(df_active) >= 5:
        fig, ax = plt.subplots()
        vals = df_active["move_ema"].values.astype(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=35)
        ax.set_xlabel("move_ema")
        ax.set_title(f"{stem} active-only move_ema distribution")
        save_plot(out_dir / f"{stem}_plot_hist_move_ema.png", fig)

    # Windowed decay plots
    if decay_df is not None and len(decay_df):
        x = 0.5 * (decay_df["t_start"].values.astype(float) + decay_df["t_end"].values.astype(float))

        fig, ax = plt.subplots()
        ax.plot(x, decay_df["sloppiness_index"].values.astype(float), label="sloppiness_index")
        ax.plot(x, decay_df["stability_index"].values.astype(float), label="stability_index", alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{stem} decay indices (windows)")
        ax.legend()
        save_plot(out_dir / f"{stem}_plot_decay_indices.png", fig)

        fig, ax = plt.subplots()
        ax.plot(x, decay_df["valid_frame_fraction"].values.astype(float), label="valid_frame_fraction")
        ax.axhline(MIN_VALID_FRAC_FOR_DECAY, linestyle="--", label="min_valid_for_decay")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{stem} window validity")
        ax.legend()
        save_plot(out_dir / f"{stem}_plot_window_validity.png", fig)

        fig, ax = plt.subplots()
        ax.plot(x, decay_df["mean_move_ema"].values.astype(float), label="mean_move_ema")
        ax.plot(x, decay_df["mean_move_jerk"].values.astype(float), label="mean_jerk", alpha=0.85)
        ax.plot(x, decay_df["mean_move_variability"].values.astype(float), label="mean_variability", alpha=0.85)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{stem} window means")
        ax.legend()
        save_plot(out_dir / f"{stem}_plot_window_means.png", fig)


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Climbing analysis using YOLOv8 Pose (v6): tracking + sanity gating + torso-first movement + richer plots."
    )
    parser.add_argument("video", help="Path to video file (mp4/mov)")
    parser.add_argument("--out", default="outputs", help="Output folder")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--proxy-width", type=int, default=PROXY_WIDTH)
    parser.add_argument("--proxy-fps", type=int, default=PROXY_FPS)

    parser.add_argument("--no-crop", action="store_true", help="Disable saving cropped video")
    parser.add_argument("--model", default=YOLO_MODEL_NAME)
    parser.add_argument("--device", default="", help="mps or cpu")

    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-h264", action="store_true")
    parser.add_argument("--t-start", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--no-bbox", action="store_true")

    args = parser.parse_args()
    DEBUG = bool(args.debug)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_proxy:
        AUTO_PROXY = False
    else:
        PROXY_WIDTH = int(args.proxy_width)
        PROXY_FPS = int(args.proxy_fps)

    YOLO_MODEL_NAME = str(args.model)

    device = str(args.device).strip().lower()
    if device == "":
        device = pick_device()

    # Proxy
    video_used = make_or_use_proxy(args.video, out_dir)
    if video_used != args.video:
        print(f"[proxy] Using: {video_used}")

    print(f"[yolo] Loading {YOLO_MODEL_NAME} on device={device} ...")
    model = load_yolo_pose(YOLO_MODEL_NAME)

    # Crop (saved)
    analysis_video = video_used
    if not args.no_crop and AUTO_CROP:
        print("[crop] Building + saving cropped video ...")
        crop_video = build_and_save_cropped_video(video_used, out_dir, model, device=device)
        print(f"[crop] Using cropped video: {crop_video}")
        analysis_video = crop_video
    else:
        print("[crop] Disabled; using original/proxy video.")

    # Analyze
    t0 = time.time()
    summary, df, df_active, pause_segments, decay_df, fps_in = analyze(
        analysis_video, model, device=device, t_start=args.t_start, t_end=args.t_end
    )
    print(summary)
    print(f"[done] Analysis time: {time.time() - t0:.2f}s")

    stem = Path(analysis_video).stem
    csv_path = out_dir / f"{stem}_timeseries_v6.csv"
    csv_active_path = out_dir / f"{stem}_active_only_v6.csv"
    decay_path = out_dir / f"{stem}_decay_windows_v6.csv"
    summary_path = out_dir / f"{stem}_summary_v6.csv"

    df.to_csv(csv_path, index=False)
    df_active.to_csv(csv_active_path, index=False)
    decay_df.to_csv(decay_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(f"Saved timeseries CSV: {csv_path}")
    print(f"Saved active-only CSV: {csv_active_path}")
    print(f"Saved decay windows CSV: {decay_path}")
    print(f"Saved summary CSV: {summary_path}")

    make_plots(out_dir, stem, df, df_active, decay_df)
    print(f"Saved plots: {out_dir}/{stem}_plot_*.png")

    # Telemetry
    if args.telemetry:
        telemetry_raw = out_dir / f"{stem}_telemetry_v6.avi"
        out_vid, fps_out = write_telemetry(
            analysis_video,
            str(telemetry_raw),
            model, device=device,
            draw_bbox=(not args.no_bbox),
            t_start=args.t_start, t_end=args.t_end
        )
        print(f"Saved telemetry (raw AVI): {telemetry_raw} (fps_out≈{fps_out:.2f})")

        if args.telemetry_h264:
            telemetry_h264 = out_dir / f"{stem}_telemetry_v6_h264.mp4"
            ffmpeg_reencode_to_h264_qt(
                str(telemetry_raw),
                str(telemetry_h264),
                fps=max(1, int(round(fps_out))),
                max_h=1920
            )
            print(f"Saved telemetry (QuickTime H.264): {telemetry_h264}")