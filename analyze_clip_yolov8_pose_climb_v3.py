
#(venv) alex@Alexs-MacBook-Pro climb_analytics % python3 analyze_clip_yolov8_pose_climb_v3.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264

#(venv) alex@Alexs-MacBook-Pro climb_analytics %  python3 analyze_clip_yolov8_pose_climb_v3.py climbing_clip_1.mp4 --out outputs --telemetry --telemetry-h264 
#more accurate?

# analyze_clip_yolov8_pose_climb_v3.py
# YOLOv8 Pose (Ultralytics) climbing analysis:
# - proxy conversion (optional)
# - pose tracking (person selection by largest bbox)
# - body-relative movement score (translation-invariant)
# - auto active-climb segmentation (filters walking-in + post-fall)
# - micro-pause detection only during active climbing
# - telemetry annotated video
# - windowed "decay" features CSV

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

# Choose model size:
#   yolov8n-pose.pt = fastest
#   yolov8s-pose.pt = more accurate
YOLO_MODEL_NAME = "yolov8n-pose.pt"

# Output sampling
ANALYSIS_STRIDE = 2              # per-frame analysis step (2 = half-rate)
TELEMETRY_STRIDE_SLOW = 4
TELEMETRY_STRIDE_FAST = 1

# Pose confidence thresholds
KPT_CONF_THR = 0.15              # consider joint usable above this
MIN_GOOD_KPTS = 6

# Movement score (body-relative) and micro-pauses
MOVEMENT_THRESHOLD = 0.030       # tune after you look at plot; lower = more "moving"
MIN_PAUSE_SECONDS = 1.25
MERGE_GAP_SECONDS = 0.35

# Adaptive stride trigger (telemetry only)
FAST_STRIDE_EMA_THR = 0.050
MOVEMENT_EMA_ALPHA = 0.25

# Active climb segmentation heuristic (state machine)
ACTIVE_START_HITS = 6            # require N "active" frames to start
ACTIVE_END_MISSES = 10           # require N "inactive" frames to end
ACTIVE_MIN_BBOX_AREA_FRAC = 0.015  # bbox area fraction of frame
ACTIVE_MAX_ROOT_Y = 0.88           # root (mid-hip) must be above this (normalized y)
ACTIVE_MIN_HANDS_ABOVE_HIPS_FRAC = 0.25  # fraction of frames where at least 1 wrist above hips in a small window
ACTIVE_HANDS_WINDOW_S = 1.0

# Decay features
WINDOW_SECONDS = 30.0
WINDOW_STEP_SECONDS = 10.0

# Telemetry / drawing
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


# -----------------------
# ffmpeg helpers
# -----------------------
def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def ffmpeg_make_proxy(in_path: str, out_path: str, width=1280, fps=30):
    """
    H.264 proxy, yuv420p, CFR fps, no audio. Does NOT modify original.
    """
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
        "-movflags", "+faststart",
        "-video_track_timescale", "600",
        "-an",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# -----------------------
# Video helpers
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
# YOLOv8 Pose backend
# -----------------------
def pick_device() -> str:
    """
    Prefer MPS on Apple Silicon if available; else CPU.
    """
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_yolo_pose(model_name: str, device: str):
    from ultralytics import YOLO
    model = YOLO(model_name)
    # warm-up inference is optional; we do it on first call anyway
    return model


def run_yolo_pose(model, frame_bgr, device: str):
    """
    Returns (kpts_yx_conf, bbox_xyxy, bbox_area_frac, has_pose)
    - kpts_yx_conf shape (17,3), normalized to full frame, format [y,x,conf]
    - bbox_xyxy in pixels (x1,y1,x2,y2)
    """
    H, W = frame_bgr.shape[:2]

    # Ultralytics expects BGR numpy arrays fine
    results = model.predict(
        source=frame_bgr,
        device=device,
        verbose=False,
        conf=0.25,          # detection conf
        iou=0.5,
        imgsz=640
    )
    if not results or len(results) == 0:
        return None, None, 0.0, False

    r = results[0]
    if r.boxes is None or r.keypoints is None:
        return None, None, 0.0, False

    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else np.array(r.boxes.xyxy)
    kpts_xyn = r.keypoints.xyn  # normalized x,y
    kpts_conf = r.keypoints.conf

    if kpts_xyn is None or kpts_conf is None:
        return None, None, 0.0, False

    kpts_xyn = kpts_xyn.cpu().numpy() if hasattr(kpts_xyn, "cpu") else np.array(kpts_xyn)
    kpts_conf = kpts_conf.cpu().numpy() if hasattr(kpts_conf, "cpu") else np.array(kpts_conf)

    if boxes.shape[0] == 0 or kpts_xyn.shape[0] == 0:
        return None, None, 0.0, False

    # pick person with largest bbox area
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

    # COCO order is 17 kpts; xyn is [x,y] normalized
    xy = kpts_xyn[idx]            # (17,2) [x,y]
    cf = kpts_conf[idx]           # (17,)
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[:, 0] = xy[:, 1]         # y
    kpts[:, 1] = xy[:, 0]         # x
    kpts[:, 2] = cf

    return kpts, bb, area_frac, True


# -----------------------
# Movement + segmentation
# -----------------------
def mid_hip_root(kpts_yx_conf):
    # COCO hips: 11, 12
    y = 0.5 * (float(kpts_yx_conf[11, 0]) + float(kpts_yx_conf[12, 0]))
    x = 0.5 * (float(kpts_yx_conf[11, 1]) + float(kpts_yx_conf[12, 1]))
    return np.array([y, x], dtype=np.float32)


def hands_above_hips(kpts):
    # wrists: 9,10 ; hips: 11,12
    lw_y = float(kpts[9, 0]);  rw_y = float(kpts[10, 0])
    lh_y = float(kpts[11, 0]); rh_y = float(kpts[12, 0])
    hips_y = 0.5 * (lh_y + rh_y)
    # smaller y = higher in image
    return (lw_y < hips_y) or (rw_y < hips_y)


def movement_body_relative(prev_kpts, curr_kpts, conf_thr=KPT_CONF_THR):
    """
    Translation-invariant movement:
      - subtract mid-hip root in each frame
      - compute mean joint delta over overlapping confident joints
    """
    if prev_kpts is None or curr_kpts is None:
        return 0.0, 0

    prev_conf = prev_kpts[:, 2]
    curr_conf = curr_kpts[:, 2]
    good = (prev_conf >= conf_thr) & (curr_conf >= conf_thr)

    n = int(good.sum())
    if n < MIN_GOOD_KPTS:
        return 0.0, n

    prev_root = mid_hip_root(prev_kpts)
    curr_root = mid_hip_root(curr_kpts)

    prev_rel = prev_kpts[:, :2] - prev_root[None, :]
    curr_rel = curr_kpts[:, :2] - curr_root[None, :]

    diffs = curr_rel[good] - prev_rel[good]
    score = float(np.linalg.norm(diffs, axis=1).mean())
    return score, n


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
            end = t
            if start is not None:
                segs.append((float(start), float(end)))
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
    """
    Simple state machine:
    - active candidate if:
        pose exists,
        bbox area above threshold (you're actually in frame),
        root_y above floor-ish threshold (not on ground),
        and in the recent window, hands-above-hips happens some fraction
    """
    n = len(times)
    active = np.zeros(n, dtype=bool)

    # rolling hands-above-hips fraction
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
            # active: require sustained miss to turn off
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
# Analysis + decay features
# -----------------------
def compute_decay_windows(df_active, window_s, step_s):
    """
    df_active is already filtered to active-climb frames only.
    Outputs windows with:
      - mean movement
      - pause fraction
      - micro-pause count (within window)
      - "work rate" proxy: movement EMA mean
    """
    if len(df_active) == 0:
        return pd.DataFrame()

    t = df_active["t"].values.astype(float)
    t0 = float(t.min())
    t1 = float(t.max())

    rows = []
    start = t0
    while start + window_s <= t1 + 1e-6:
        end = start + window_s
        win = df_active[(df_active["t"] >= start) & (df_active["t"] < end)]
        if len(win) >= 3:
            mean_move = float(win["move_body"].mean())
            pause_frac = float((win["is_paused"] == True).mean())
            mean_ema = float(win["move_ema"].mean())

            # micro-pause count inside the window from pause segments
            segs = _segments_from_boolean(win["t"].values, win["is_paused"].values)
            segs = _filter_and_merge_segments(segs, MIN_PAUSE_SECONDS, MERGE_GAP_SECONDS)
            mp_count = int(len(segs))
            mp_time = float(sum(e - s for s, e in segs))

            rows.append({
                "t_start": float(start),
                "t_end": float(end),
                "mean_move_body": mean_move,
                "mean_move_ema": mean_ema,
                "pause_fraction": pause_frac,
                "micro_pause_count": mp_count,
                "micro_pause_time_s": mp_time,
            })

        start += step_s

    return pd.DataFrame(rows)


def analyze(video_path: str, model, device: str, t_start: float | None, t_end: float | None):
    cap = open_video_or_raise(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1:
        fps = 30.0

    times = []
    has_pose = []
    bbox_area = []
    root_y = []
    hands_flag = []
    move_body = []
    move_ema = []

    prev_kpts = None
    ema = 0.0

    frame_idx = 0
    for frame in raw_frames(cap):
        t = frame_idx / fps

        if t_start is not None and t < t_start:
            frame_idx += 1
            continue
        if t_end is not None and t > t_end:
            break

        if frame_idx % ANALYSIS_STRIDE != 0:
            frame_idx += 1
            continue

        kpts, bb, area_frac, ok = run_yolo_pose(model, frame, device=device)
        if not ok or kpts is None:
            times.append(float(t))
            has_pose.append(False)
            bbox_area.append(0.0)
            root_y.append(1.0)
            hands_flag.append(False)
            move_body.append(0.0)
            ema = (1 - MOVEMENT_EMA_ALPHA) * ema
            move_ema.append(float(ema))
            prev_kpts = None
            frame_idx += 1
            continue

        # root + hands flags even if some joints are weak
        r = mid_hip_root(kpts)
        hy = hands_above_hips(kpts)

        score, n_good = movement_body_relative(prev_kpts, kpts, conf_thr=KPT_CONF_THR)
        prev_kpts = kpts

        ema = MOVEMENT_EMA_ALPHA * score + (1 - MOVEMENT_EMA_ALPHA) * ema

        times.append(float(t))
        has_pose.append(True)
        bbox_area.append(float(area_frac))
        root_y.append(float(r[0]))
        hands_flag.append(bool(hy))
        move_body.append(float(score))
        move_ema.append(float(ema))

        frame_idx += 1

    cap.release()

    df = pd.DataFrame({
        "t": times,
        "has_pose": has_pose,
        "bbox_area_frac": bbox_area,
        "root_y": root_y,
        "hands_above_hips": hands_flag,
        "move_body": move_body,
        "move_ema": move_ema,
    })

    fps_eff = fps / max(1, ANALYSIS_STRIDE)

    # active climb mask
    active = compute_active_climb_mask(
        df["t"].values,
        df["has_pose"].values,
        df["root_y"].values,
        df["bbox_area_frac"].values,
        df["hands_above_hips"].values,
        fps_eff=fps_eff
    )
    df["active"] = active

    # pause logic should be computed ONLY during active climbing
    df["is_moving"] = False
    df["is_paused"] = False

    act = df["active"].values.astype(bool)
    mv = df["move_body"].values.astype(float)

    df.loc[act, "is_moving"] = mv[act] >= MOVEMENT_THRESHOLD
    df.loc[act, "is_paused"] = mv[act] < MOVEMENT_THRESHOLD

    # micro-pause segments only inside active region
    df_active = df[df["active"] == True].copy()

    segs = _segments_from_boolean(df_active["t"].values, df_active["is_paused"].values)
    pause_segments = _filter_and_merge_segments(segs, MIN_PAUSE_SECONDS, MERGE_GAP_SECONDS)

    duration = float(df["t"].iloc[-1]) if len(df) else 0.0
    active_duration = float(df_active["t"].iloc[-1] - df_active["t"].iloc[0]) if len(df_active) >= 2 else 0.0
    pause_time = float(sum(e - s for s, e in pause_segments))
    pause_count = int(len(pause_segments))
    move_time = float(max(0.0, active_duration - pause_time))

    summary = {
        "video": Path(video_path).name,
        "duration_s": round(duration, 2),
        "active_climb_duration_s": round(active_duration, 2),
        "micro_pause_time_s": round(pause_time, 2),
        "micro_pause_count": pause_count,
        "active_move_time_s": round(move_time, 2),
        "avg_move_body_active": round(float(df_active["move_body"].mean()) if len(df_active) else 0.0, 6),
        "pose_detect_rate": round(float(df["has_pose"].mean()) if len(df) else 0.0, 3),
        "analysis_fps": round(float(fps_eff), 2),
        "device": device,
        "model": YOLO_MODEL_NAME,
    }

    # decay windows from active frames
    decay_df = compute_decay_windows(df_active, WINDOW_SECONDS, WINDOW_STEP_SECONDS)

    return summary, df, df_active, pause_segments, decay_df, fps


# -----------------------
# Telemetry video
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

    # simple active mask state for telemetry (same heuristic)
    active_on = False
    hits = 0
    misses = 0

    # hands rolling
    hands_win = max(1, int(round(ACTIVE_HANDS_WINDOW_S * fps_out)))
    hands_hist = []

    ema = 0.0
    prev_kpts = None

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

        kpts, bb, area_frac, ok_pose = run_yolo_pose(model, frame, device=device)

        vis = frame.copy()
        has_pose = bool(ok_pose and kpts is not None)

        move = 0.0
        rooty = 1.0
        hands = False

        if has_pose:
            root = mid_hip_root(kpts)
            rooty = float(root[0])
            hands = hands_above_hips(kpts)

            move, n_good = movement_body_relative(prev_kpts, kpts, conf_thr=KPT_CONF_THR)
            prev_kpts = kpts

            ema = MOVEMENT_EMA_ALPHA * move + (1 - MOVEMENT_EMA_ALPHA) * ema

            if draw_bbox and bb is not None:
                x1, y1, x2, y2 = bb
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 2)

            draw_pose(vis, kpts, conf_thr=DRAW_CONF_THRESHOLD)
        else:
            prev_kpts = None
            ema = (1 - MOVEMENT_EMA_ALPHA) * ema

        # hands rolling fraction
        hands_hist.append(1 if hands else 0)
        if len(hands_hist) > hands_win:
            hands_hist = hands_hist[-hands_win:]
        hands_frac = float(sum(hands_hist)) / float(max(1, len(hands_hist)))

        # active candidate
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
                    # reset pause state when not active
                    pause_started_t = None
                    confirmed_pause = False

        # micro-pause logic only while active
        if active_on and has_pose:
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

        history.append(float(ema))
        if len(history) > window_len:
            history = history[-window_len:]

        # panel
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        if not has_pose:
            state = "NO POSE"
            color = (0, 0, 255)
        else:
            if not active_on:
                state = "INACTIVE"
                color = (200, 200, 200)
            else:
                if ema >= MOVEMENT_THRESHOLD:
                    state = "MOVING"
                    color = (0, 255, 0)
                else:
                    state = "PAUSING" if not confirmed_pause else "PAUSE"
                    color = (0, 165, 255) if not confirmed_pause else (255, 0, 255)

        lines = [
            (f"time: {t:6.2f}s", (240, 240, 240)),
            (f"state: {state}", color),
            (f"move body: {move:.4f}", (240, 240, 240)),
            (f"move EMA: {ema:.4f}", (240, 240, 240)),
            (f"stride: {stride_now}", (200, 200, 200)),
            (f"active: {'yes' if active_on else 'no'}", (240, 240, 240)),
            (f"bbox area: {float(area_frac):.3f}", (200, 200, 200)),
            (f"root_y: {rooty:.3f}", (200, 200, 200)),
            (f"hands frac: {hands_frac:.2f}", (200, 200, 200)),
            (f"micro-pauses: {micro_pause_count}", (240, 240, 240)),
            (f"micro-pause time: {micro_pause_time:5.1f}s", (240, 240, 240)),
            (f"device: {device}", (240, 240, 240)),
            (f"model: {YOLO_MODEL_NAME}", (240, 240, 240)),
        ]
        draw_panel(panel, lines, header="Climb Telemetry (YOLO Pose)")
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
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Climbing analysis using YOLOv8 Pose: body-relative movement, active climb segmentation, decay features."
    )
    parser.add_argument("video", help="Path to video file (mp4/mov)")
    parser.add_argument("--out", default="outputs", help="Output folder (default: ./outputs)")
    parser.add_argument("--debug", action="store_true", help="Debug prints")
    parser.add_argument("--no-proxy", action="store_true", help="Disable auto proxy conversion")
    parser.add_argument("--proxy-width", type=int, default=PROXY_WIDTH)
    parser.add_argument("--proxy-fps", type=int, default=PROXY_FPS)

    parser.add_argument("--model", default=YOLO_MODEL_NAME, help="YOLO pose model (e.g. yolov8n-pose.pt, yolov8s-pose.pt)")
    parser.add_argument("--device", default="", help="Override device: mps or cpu")

    parser.add_argument("--telemetry", action="store_true", help="Write telemetry AVI")
    parser.add_argument("--telemetry-h264", action="store_true", help="Also write QuickTime-friendly H.264 MP4 (requires ffmpeg)")
    parser.add_argument("--t-start", type=float, default=None, help="Optional analysis start time (seconds)")
    parser.add_argument("--t-end", type=float, default=None, help="Optional analysis end time (seconds)")
    parser.add_argument("--no-bbox", action="store_true", help="Do not draw bbox in telemetry")

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

    video_used = make_or_use_proxy(args.video, out_dir)
    if video_used != args.video:
        print(f"[proxy] Using: {video_used}")

    print(f"[yolo] Loading {YOLO_MODEL_NAME} on device={device} ...")
    model = load_yolo_pose(YOLO_MODEL_NAME, device=device)

    t0 = time.time()
    summary, df, df_active, pause_segments, decay_df, fps_in = analyze(
        video_used, model, device=device, t_start=args.t_start, t_end=args.t_end
    )
    print(summary)
    print(f"[done] Analysis time: {time.time() - t0:.2f}s")

    stem = Path(video_used).stem
    csv_path = out_dir / f"{stem}_yolo_pose_timeseries.csv"
    csv_active_path = out_dir / f"{stem}_yolo_pose_active_only.csv"
    decay_path = out_dir / f"{stem}_decay_windows.csv"
    png_path = out_dir / f"{stem}_movement_plot.png"

    df.to_csv(csv_path, index=False)
    df_active.to_csv(csv_active_path, index=False)
    decay_df.to_csv(decay_path, index=False)

    # Plot movement with active mask overlay
    plt.figure()
    plt.plot(df["t"], df["move_body"], label="move_body")
    plt.plot(df["t"], df["move_ema"], label="move_ema", alpha=0.8)
    plt.axhline(MOVEMENT_THRESHOLD, linestyle="--", label="thr")
    # shade active region
    active = df["active"].values.astype(bool)
    if active.any():
        segs = _segments_from_boolean(df["t"].values, active)
        for s, e in segs:
            plt.axvspan(s, e, alpha=0.15)
    plt.xlabel("Time (s)")
    plt.ylabel("Body-relative movement")
    plt.title(f"{stem} movement (active shaded)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print(f"Saved timeseries CSV: {csv_path}")
    print(f"Saved active-only CSV: {csv_active_path}")
    print(f"Saved decay windows CSV: {decay_path}")
    print(f"Saved plot: {png_path}")

    if args.telemetry:
        telemetry_raw = out_dir / f"{stem}_telemetry_yolo.avi"
        out_vid, fps_out = write_telemetry(
            video_used,
            str(telemetry_raw),
            model, device=device,
            draw_bbox=(not args.no_bbox),
            t_start=args.t_start, t_end=args.t_end
        )
        print(f"Saved telemetry (raw AVI): {telemetry_raw} (fps_out≈{fps_out:.2f})")

        if args.telemetry_h264:
            telemetry_h264 = out_dir / f"{stem}_telemetry_yolo_h264.mp4"
            ffmpeg_reencode_to_h264_qt(str(telemetry_raw), str(telemetry_h264), fps=max(1, int(round(fps_out))), max_h=1920)
            print(f"Saved telemetry (QuickTime H.264): {telemetry_h264}")