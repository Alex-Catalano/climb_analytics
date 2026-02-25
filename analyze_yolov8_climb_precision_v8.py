#!/usr/bin/env python3
"""
YOLO pose climb analyzer v8 (accuracy-focused rewrite).

Key upgrades versus earlier versions:
- Deterministic target locking without relying on tracker IDs.
- ROI-focused re-detection to keep keypoint resolution high.
- Per-joint Kalman smoothing with confidence-aware gating.
- Robust movement metric (confidence/trust weighted, outlier-clipped).
- Annotated visuals by default (H.264 when ffmpeg is available).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl-cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "ultralytics is required. Activate your venv and install dependencies first."
    ) from exc


# COCO-17 skeleton edges.
SKELETON_EDGES: list[tuple[int, int]] = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

JOINT_SPEED_WEIGHTS = np.array(
    [
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.70,
        0.70,
        0.85,
        0.85,
        1.00,
        1.00,
        0.60,
        0.60,
        0.90,
        0.90,
        1.00,
        1.00,
    ],
    dtype=np.float32,
)

POSE_QUALITY_WEIGHTS = np.array(
    [
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        1.00,
        1.00,
        0.90,
        0.90,
        1.00,
        1.00,
        1.00,
        1.00,
        0.90,
        0.90,
        1.00,
        1.00,
    ],
    dtype=np.float32,
)

TORSO_PAIRS: list[tuple[int, int]] = [
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 12),
    (6, 11),
]

QUALITY_PRESETS = {
    "fast": {
        "imgsz": 832,
        "conf": 0.12,
        "analysis_stride": 2,
        "ema_alpha": 0.35,
        "move_on": 0.080,
        "move_off": 0.055,
    },
    "balanced": {
        "imgsz": 960,
        "conf": 0.08,
        "analysis_stride": 1,
        "ema_alpha": 0.28,
        "move_on": 0.070,
        "move_off": 0.048,
    },
    "max": {
        "imgsz": 1280,
        "conf": 0.05,
        "analysis_stride": 1,
        "ema_alpha": 0.22,
        "move_on": 0.062,
        "move_off": 0.043,
    },
}


@dataclass
class PoseCandidate:
    bbox_xyxy: np.ndarray
    det_conf: float
    keypoints_xy: np.ndarray  # (17,2) pixel
    keypoints_conf: np.ndarray  # (17,)
    pose_quality: float
    area_frac: float


class JointKalman2D:
    """Constant-velocity Kalman filter for one keypoint."""

    def __init__(self) -> None:
        self.initialized = False
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.p = np.eye(4, dtype=np.float32) * 1e3

    def initialize(self, px: float, py: float) -> None:
        self.initialized = True
        self.x[:] = 0.0
        self.x[0, 0] = float(px)
        self.x[1, 0] = float(py)
        self.p = np.eye(4, dtype=np.float32) * 20.0

    def predict(self, dt: float) -> None:
        if not self.initialized:
            return
        dt = max(1e-3, float(dt))
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q_pos = 18.0
        q_vel = 18.0
        q = np.array(
            [
                [0.25 * dt4 * q_pos, 0.0, 0.5 * dt3 * q_pos, 0.0],
                [0.0, 0.25 * dt4 * q_pos, 0.0, 0.5 * dt3 * q_pos],
                [0.5 * dt3 * q_vel, 0.0, dt2 * q_vel, 0.0],
                [0.0, 0.5 * dt3 * q_vel, 0.0, dt2 * q_vel],
            ],
            dtype=np.float32,
        )
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def update(self, z_xy: np.ndarray, meas_var: float, gate_thresh: float) -> bool:
        if not self.initialized:
            return False
        z = z_xy.reshape(2, 1).astype(np.float32)
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        r = np.eye(2, dtype=np.float32) * float(meas_var)
        y = z - (h @ self.x)
        s = h @ self.p @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return False
        d2 = float((y.T @ s_inv @ y).item())
        if d2 > float(gate_thresh):
            return False
        k = self.p @ h.T @ s_inv
        i = np.eye(4, dtype=np.float32)
        self.x = self.x + k @ y
        self.p = (i - k @ h) @ self.p
        return True

    @property
    def position(self) -> np.ndarray:
        if not self.initialized:
            return np.array([np.nan, np.nan], dtype=np.float32)
        return self.x[:2, 0].astype(np.float32)


class PoseSmoother:
    def __init__(
        self,
        n_joints: int = 17,
        conf_thr: float = 0.20,
        median_window: int = 3,
        max_predict_gap: int = 1,
        reset_after_gap: int = 6,
    ) -> None:
        self.n_joints = int(n_joints)
        self.conf_thr = float(conf_thr)
        self.filters = [JointKalman2D() for _ in range(self.n_joints)]
        self.trust = np.zeros(self.n_joints, dtype=np.float32)
        self.history = deque(maxlen=max(1, int(median_window)))
        self.max_predict_gap = max(0, int(max_predict_gap))
        self.reset_after_gap = max(self.max_predict_gap + 1, int(reset_after_gap))
        self.frames_since_seen = np.full(self.n_joints, self.reset_after_gap + 1, dtype=np.int32)

    def reset(self) -> None:
        for f in self.filters:
            f.initialized = False
            f.x[:] = 0.0
            f.p = np.eye(4, dtype=np.float32) * 1e3
        self.trust[:] = 0.0
        self.frames_since_seen[:] = self.reset_after_gap + 1
        self.history.clear()

    def step(
        self,
        keypoints_xy: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if keypoints_xy is None:
            keypoints_xy = np.full((self.n_joints, 2), np.nan, dtype=np.float32)
        if keypoints_conf is None:
            keypoints_conf = np.zeros(self.n_joints, dtype=np.float32)

        keypoints_xy = keypoints_xy.astype(np.float32, copy=False)
        keypoints_conf = keypoints_conf.astype(np.float32, copy=False)

        pose = np.full((self.n_joints, 2), np.nan, dtype=np.float32)

        for j in range(self.n_joints):
            f = self.filters[j]
            f.predict(dt)

            x, y = keypoints_xy[j]
            conf = float(keypoints_conf[j])

            has_meas = conf >= self.conf_thr and np.isfinite(x) and np.isfinite(y)

            if has_meas:
                if not f.initialized:
                    f.initialize(float(x), float(y))
                    self.trust[j] = 1.0
                    self.frames_since_seen[j] = 0
                else:
                    meas_var = max(2.0, 2.0 + 30.0 * ((1.0 - conf) ** 2))
                    gate = 20.0 if conf >= 0.75 else 14.0
                    accepted = f.update(np.array([x, y], dtype=np.float32), meas_var=meas_var, gate_thresh=gate)
                    if accepted:
                        self.trust[j] = min(1.0, self.trust[j] * 0.6 + 0.55)
                        self.frames_since_seen[j] = 0
                    elif conf >= 0.90:
                        # Strong measurements can reset stale states.
                        f.initialize(float(x), float(y))
                        self.trust[j] = 0.9
                        self.frames_since_seen[j] = 0
                    else:
                        self.trust[j] *= 0.75
                        self.frames_since_seen[j] += 1
            else:
                self.trust[j] *= 0.75
                self.frames_since_seen[j] += 1

            if self.frames_since_seen[j] > self.max_predict_gap:
                pose[j] = np.array([np.nan, np.nan], dtype=np.float32)
                if self.frames_since_seen[j] > self.reset_after_gap:
                    f.initialized = False
                    self.trust[j] = 0.0
            else:
                pose[j] = f.position

        age = self.frames_since_seen.copy()

        self.history.append(pose.copy())
        if len(self.history) >= 3:
            stacked = np.stack(self.history, axis=0)
            pose_med = pose.copy()
            for j in range(self.n_joints):
                for axis in range(2):
                    vals = stacked[:, j, axis]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        pose_med[j, axis] = float(np.median(vals))
            pose = pose_med
        for j in range(self.n_joints):
            if age[j] > self.max_predict_gap:
                pose[j] = np.array([np.nan, np.nan], dtype=np.float32)
                self.trust[j] = 0.0

        return pose, self.trust.copy(), age


class TargetLock:
    """Single-climber target continuity with score-based matching."""

    def __init__(self, max_misses: int = 45) -> None:
        self.max_misses = int(max_misses)
        self.current_track_id = 1
        self.next_track_id = 2
        self.id_switches = 0
        self.misses = 0
        self.last_bbox: np.ndarray | None = None
        self.last_center: np.ndarray | None = None

    def select(self, candidates: list[PoseCandidate], frame_shape: tuple[int, int, int]) -> tuple[int | None, int | None]:
        if not candidates:
            self.misses += 1
            if self.misses > self.max_misses:
                self.last_bbox = None
                self.last_center = None
            return None, None

        h, w = frame_shape[:2]
        diag = float(math.hypot(w, h))
        best_idx = None
        best_score = -1e9

        for idx, c in enumerate(candidates):
            score = 0.45 * c.pose_quality + 0.15 * c.det_conf + 0.10 * min(1.0, c.area_frac * 8.0)

            cx = 0.5 * (float(c.bbox_xyxy[0]) + float(c.bbox_xyxy[2]))
            cy = 0.5 * (float(c.bbox_xyxy[1]) + float(c.bbox_xyxy[3]))
            center = np.array([cx, cy], dtype=np.float32)

            if self.last_bbox is not None:
                iou = bbox_iou(c.bbox_xyxy, self.last_bbox)
                score += 0.20 * iou
            if self.last_center is not None:
                dist = float(np.linalg.norm(center - self.last_center))
                center_score = 1.0 - min(1.0, dist / (0.40 * diag + 1e-6))
                score += 0.20 * center_score
            else:
                dist = float(np.linalg.norm(center - np.array([w * 0.5, h * 0.5], dtype=np.float32)))
                score += 0.10 * (1.0 - min(1.0, dist / (0.65 * diag + 1e-6)))

            if score > best_score:
                best_score = score
                best_idx = idx

        assert best_idx is not None
        best = candidates[best_idx]
        best_center = np.array(
            [0.5 * (best.bbox_xyxy[0] + best.bbox_xyxy[2]), 0.5 * (best.bbox_xyxy[1] + best.bbox_xyxy[3])],
            dtype=np.float32,
        )

        if self.last_bbox is not None and self.last_center is not None:
            iou = bbox_iou(best.bbox_xyxy, self.last_bbox)
            center_jump = float(np.linalg.norm(best_center - self.last_center) / (diag + 1e-6))
            if self.misses >= 8 and iou < 0.04 and center_jump > 0.22:
                self.current_track_id = self.next_track_id
                self.next_track_id += 1
                self.id_switches += 1

        self.last_bbox = best.bbox_xyxy.copy()
        self.last_center = best_center
        self.misses = 0
        return best_idx, self.current_track_id

    def roi_box(self, frame_w: int, frame_h: int, pad: float = 1.35) -> tuple[int, int, int, int] | None:
        if self.last_bbox is None:
            return None
        x1, y1, x2, y2 = self.last_bbox.astype(np.float32)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bw = max(64.0, (x2 - x1) * float(pad))
        bh = max(64.0, (y2 - y1) * float(pad))
        nx1 = int(max(0, math.floor(cx - 0.5 * bw)))
        ny1 = int(max(0, math.floor(cy - 0.5 * bh)))
        nx2 = int(min(frame_w, math.ceil(cx + 0.5 * bw)))
        ny2 = int(min(frame_h, math.ceil(cy + 0.5 * bh)))
        if nx2 - nx1 < 64 or ny2 - ny1 < 64:
            return None
        return nx1, ny1, nx2, ny2


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def estimate_torso_px(pose_xy: np.ndarray) -> float:
    dists: list[float] = []
    for a, b in TORSO_PAIRS:
        pa = pose_xy[a]
        pb = pose_xy[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            d = float(np.linalg.norm(pb - pa))
            if d > 1e-6:
                dists.append(d)
    if not dists:
        return float("nan")
    return float(np.median(np.array(dists, dtype=np.float32)))


def estimate_torso_px_weighted(pose_xy: np.ndarray, conf: np.ndarray, conf_thr: float = 0.22) -> float:
    dists: list[float] = []
    for a, b in TORSO_PAIRS:
        if float(conf[a]) < conf_thr or float(conf[b]) < conf_thr:
            continue
        pa = pose_xy[a]
        pb = pose_xy[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            d = float(np.linalg.norm(pb - pa))
            if d > 1e-6:
                dists.append(d)
    if not dists:
        return float("nan")
    return float(np.median(np.array(dists, dtype=np.float32)))


def compute_pose_quality(conf: np.ndarray) -> float:
    conf = np.clip(np.asarray(conf, dtype=np.float32), 0.0, 1.0)
    if conf.shape[0] != POSE_QUALITY_WEIGHTS.shape[0]:
        return float(np.nanmean(conf))
    return float(np.average(conf, weights=POSE_QUALITY_WEIGHTS))


def is_plausible_candidate(
    c: PoseCandidate,
    frame_h: int,
    min_det_conf: float,
    min_pose_quality: float,
    min_torso_px: float,
    joint_conf_thr: float = 0.22,
) -> bool:
    if not np.isfinite(c.det_conf) or c.det_conf < float(min_det_conf):
        return False
    if c.pose_quality < float(min_pose_quality):
        return False

    conf = np.asarray(c.keypoints_conf, dtype=np.float32)
    valid_count = int(np.sum(conf >= float(joint_conf_thr)))
    if valid_count < 6:
        return False

    shoulder_count = int((conf[[5, 6]] >= 0.25).sum())
    hip_count = int((conf[[11, 12]] >= 0.25).sum())
    if shoulder_count == 0 or hip_count == 0:
        return False

    x1, y1, x2, y2 = c.bbox_xyxy.astype(np.float32)
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    aspect = bh / bw
    if not (0.45 <= aspect <= 4.0):
        return False

    # Tiny detections are unstable for keypoint tracking in this setup.
    if bh < max(72.0, 0.08 * float(frame_h)):
        return False

    torso = estimate_torso_px_weighted(c.keypoints_xy, conf, conf_thr=joint_conf_thr)
    if not np.isfinite(torso):
        return False
    if torso < 0.40 * float(min_torso_px):
        return False

    # Reject clearly upside-down or wildly inconsistent torso layout.
    sh = c.keypoints_xy[[5, 6], 1]
    hp = c.keypoints_xy[[11, 12], 1]
    sh = sh[conf[[5, 6]] >= 0.25]
    hp = hp[conf[[11, 12]] >= 0.25]
    if sh.size > 0 and hp.size > 0:
        if float(np.mean(sh) - np.mean(hp)) > 0.65 * torso:
            return False

    return True


def movement_score(
    prev_pose: np.ndarray | None,
    curr_pose: np.ndarray,
    prev_trust: np.ndarray | None,
    curr_trust: np.ndarray,
    torso_px: float,
    dt: float,
) -> float:
    if prev_pose is None or prev_trust is None:
        return float("nan")
    if not np.isfinite(torso_px):
        return float("nan")

    dt = max(1e-3, float(dt))
    scale = max(1.0, float(torso_px))
    delta = curr_pose - prev_pose
    speed = np.linalg.norm(delta, axis=1) / scale / dt

    valid = np.isfinite(speed) & (prev_trust > 0.35) & (curr_trust > 0.35)
    if int(valid.sum()) < 4:
        return float("nan")

    s = speed[valid]
    w = JOINT_SPEED_WEIGHTS[valid]
    cap = float(np.quantile(s, 0.90)) if s.size > 5 else float(np.max(s))
    s = np.minimum(s, cap)
    return float(np.average(s, weights=w))


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    candidates = ["MJPG", "XVID", "mp4v"]
    for code in candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), float(fps), (int(width), int(height)))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Could not open video writer for {path}")


def ffmpeg_h264(src: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode == 0 and dst.exists()


def draw_pose_overlay(
    frame: np.ndarray,
    pose_xy: np.ndarray,
    trust: np.ndarray,
    joint_age: np.ndarray,
    moving: bool,
    move_raw: float,
    move_ema: float,
    pose_quality: float,
    track_id: int | None,
    id_switches: int,
    frame_idx: int,
    t_sec: float,
) -> np.ndarray:
    out = frame.copy()
    line_color = (64, 220, 64) if moving else (0, 200, 255)

    for a, b in SKELETON_EDGES:
        if joint_age[a] > 1 or joint_age[b] > 1:
            continue
        if trust[a] < 0.22 or trust[b] < 0.22:
            continue
        pa = pose_xy[a]
        pb = pose_xy[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), line_color, 3, cv2.LINE_AA)

    for j in range(pose_xy.shape[0]):
        if joint_age[j] > 1:
            continue
        if trust[j] < 0.18:
            continue
        p = pose_xy[j]
        if not np.isfinite(p).all():
            continue
        radius = 3 if trust[j] < 0.55 else 5
        color = (255, 180, 0) if trust[j] < 0.55 else (0, 255, 255)
        cv2.circle(out, (int(p[0]), int(p[1])), radius, color, -1, cv2.LINE_AA)

    h, w = out.shape[:2]
    hud_h = 170
    overlay = out.copy()
    cv2.rectangle(overlay, (12, 12), (min(w - 12, 540), min(h - 12, 12 + hud_h)), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.42, out, 0.58, 0)

    state_txt = "MOVING" if moving else "PAUSED"
    lines = [
        f"frame={frame_idx}  t={t_sec:7.2f}s",
        f"track={track_id if track_id is not None else '-'}  id_switches={id_switches}",
        f"pose_quality={pose_quality:.3f}",
        f"movement_raw={move_raw:.4f}",
        f"movement_ema={move_ema:.4f}  state={state_txt}",
    ]
    y = 42
    for line in lines:
        cv2.putText(out, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2, cv2.LINE_AA)
        y += 29

    bar_x1, bar_y1 = 24, min(h - 32, 12 + hud_h - 24)
    bar_w, bar_h = 280, 14
    cv2.rectangle(out, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + bar_h), (160, 160, 160), 1)
    fill = int(np.clip(move_ema / 0.14, 0.0, 1.0) * bar_w)
    fill_color = (64, 220, 64) if moving else (0, 200, 255)
    if fill > 0:
        cv2.rectangle(out, (bar_x1, bar_y1), (bar_x1 + fill, bar_y1 + bar_h), fill_color, -1)

    return out


def summarize_pauses(df: pd.DataFrame, min_pause_sec: float) -> tuple[int, float]:
    if df.empty:
        return 0, 0.0

    pauses = 0
    longest = 0.0
    pause_start = None

    for _, row in df.iterrows():
        t = float(row["time_sec"])
        moving = bool(row["is_moving"])

        if not moving and pause_start is None:
            pause_start = t
        if moving and pause_start is not None:
            dur = t - pause_start
            if dur >= min_pause_sec:
                pauses += 1
                longest = max(longest, dur)
            pause_start = None

    if pause_start is not None:
        dur = float(df["time_sec"].iloc[-1]) - pause_start
        if dur >= min_pause_sec:
            pauses += 1
            longest = max(longest, dur)

    return pauses, float(longest)


def save_plots(df: pd.DataFrame, out_dir: Path, stem: str) -> list[Path]:
    if df.empty:
        return []

    out_paths: list[Path] = []

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4.2), dpi=140)
    ax1.plot(df["time_sec"], df["movement_raw"], label="movement_raw", alpha=0.45)
    ax1.plot(df["time_sec"], df["movement_ema"], label="movement_ema", linewidth=2.0)
    ax1.set_title("Movement Score")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalized Speed")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")
    p1 = out_dir / f"{stem}_plot_movement_v8.png"
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    out_paths.append(p1)

    fig, ax2 = plt.subplots(1, 1, figsize=(10, 4.2), dpi=140)
    ax2.plot(df["time_sec"], df["pose_quality"], label="pose_quality", color="tab:purple")
    ax2.plot(df["time_sec"], df["valid_joint_count"] / 17.0, label="valid_joint_frac", color="tab:green", alpha=0.7)
    ax2.set_title("Pose Quality")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")
    p2 = out_dir / f"{stem}_plot_pose_quality_v8.png"
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    out_paths.append(p2)

    return out_paths


def build_candidates(
    result,
    frame_w: int,
    frame_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[PoseCandidate]:
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    det_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    if result.keypoints is None or len(result.keypoints) == 0:
        return []

    kxy = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    if result.keypoints.conf is not None:
        kconf = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
    else:
        kconf = np.ones((kxy.shape[0], kxy.shape[1]), dtype=np.float32)

    candidates: list[PoseCandidate] = []
    frame_area = float(max(1, frame_w * frame_h))

    n = min(boxes_xyxy.shape[0], kxy.shape[0], kconf.shape[0])
    for i in range(n):
        box = boxes_xyxy[i].copy()
        box[[0, 2]] += float(offset_x)
        box[[1, 3]] += float(offset_y)

        keypoints_xy = kxy[i].copy()
        keypoints_xy[:, 0] += float(offset_x)
        keypoints_xy[:, 1] += float(offset_y)

        keypoints_conf = kconf[i].copy()
        pq = compute_pose_quality(keypoints_conf)

        bw = max(0.0, float(box[2] - box[0]))
        bh = max(0.0, float(box[3] - box[1]))
        area_frac = (bw * bh) / frame_area

        candidates.append(
            PoseCandidate(
                bbox_xyxy=box,
                det_conf=float(det_conf[i]),
                keypoints_xy=keypoints_xy,
                keypoints_conf=keypoints_conf,
                pose_quality=float(pq),
                area_frac=float(area_frac),
            )
        )

    return candidates


def resolve_model(model_arg: str | None) -> str:
    if model_arg:
        return model_arg

    local_preferred = [
        "yolo11x-pose.pt",
        "yolo11l-pose.pt",
        "yolov8x-pose.pt",
        "yolov8l-pose.pt",
        "yolov8s-pose.pt",
        "yolov8n-pose.pt",
    ]
    for name in local_preferred:
        if Path(name).exists():
            return name

    # Fallback name lets Ultralytics auto-download if network is available.
    return "yolo11l-pose.pt"


def iter_predict(
    model: YOLO,
    frame: np.ndarray,
    args,
    roi: tuple[int, int, int, int] | None,
) -> list[PoseCandidate]:
    h, w = frame.shape[:2]
    if roi is None:
        result = model.predict(
            source=frame,
            verbose=False,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            classes=[0],
            device=args.device,
        )[0]
        return build_candidates(result, frame_w=w, frame_h=h, offset_x=0, offset_y=0)

    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    result = model.predict(
        source=crop,
        verbose=False,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        classes=[0],
        device=args.device,
    )[0]
    return build_candidates(result, frame_w=w, frame_h=h, offset_x=x1, offset_y=y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO climb analysis v8 (accuracy-first + visual telemetry)."
    )
    parser.add_argument("video", type=str, help="Input video path.")
    parser.add_argument("--out", type=str, default="outputs_v8", help="Output directory.")

    parser.add_argument("--quality", choices=["fast", "balanced", "max"], default="max")
    parser.add_argument("--model", type=str, default=None, help="Pose model path/name.")
    parser.add_argument("--device", type=str, default="", help="Ultralytics device (cpu, mps, 0, ...).")

    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=None, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=8, help="Maximum detections per frame.")

    parser.add_argument("--analysis-stride", type=int, default=None, help="Process every Nth frame.")
    parser.add_argument("--ema-alpha", type=float, default=None, help="EMA smoothing for movement score.")
    parser.add_argument("--move-on", type=float, default=None, help="Movement ON threshold.")
    parser.add_argument("--move-off", type=float, default=None, help="Movement OFF threshold.")
    parser.add_argument("--state-hold-frames", type=int, default=5, help="Hysteresis hold frames.")

    parser.add_argument(
        "--visuals",
        choices=["none", "avi", "h264"],
        default="h264",
        help="Output annotated video format.",
    )
    parser.add_argument("--keep-raw-avi", action="store_true", help="Keep intermediate AVI when --visuals=h264.")

    parser.add_argument("--roi-pad", type=float, default=1.35, help="ROI expansion around target bbox.")
    parser.add_argument("--full-refresh", type=int, default=8, help="Force full-frame detection every N processed frames.")
    parser.add_argument("--max-misses", type=int, default=45, help="Frames to keep track lock before reset.")
    parser.add_argument("--lock-min-det-conf", type=float, default=0.12, help="Minimum detector confidence for target lock.")
    parser.add_argument("--lock-min-pose-quality", type=float, default=0.18, help="Minimum pose-quality score for target lock.")
    parser.add_argument("--joint-conf-thr", type=float, default=0.22, help="Per-joint confidence threshold for smoothing/drawing.")

    parser.add_argument("--min-pause-sec", type=float, default=1.0, help="Minimum pause duration for summary.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for quick checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Input video not found: {video_path}")

    preset = QUALITY_PRESETS[args.quality]
    if args.imgsz is None:
        args.imgsz = int(preset["imgsz"])
    if args.conf is None:
        args.conf = float(preset["conf"])
    if args.analysis_stride is None:
        args.analysis_stride = int(preset["analysis_stride"])
    if args.ema_alpha is None:
        args.ema_alpha = float(preset["ema_alpha"])
    if args.move_on is None:
        args.move_on = float(preset["move_on"])
    if args.move_off is None:
        args.move_off = float(preset["move_off"])

    model_name = resolve_model(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== v8 climb analysis ===")
    print(f"video        : {video_path}")
    print(f"model        : {model_name}")
    print(f"quality      : {args.quality}")
    print(f"imgsz/conf   : {args.imgsz} / {args.conf}")
    print(f"stride       : {args.analysis_stride}")
    print(f"visuals      : {args.visuals}")

    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    min_torso_px = max(24.0, 0.02 * max(width, height))

    writer = None
    raw_visual_path: Path | None = None
    final_visual_path: Path | None = None

    stem = video_path.stem
    if args.visuals != "none":
        if args.visuals == "h264":
            raw_visual_path = out_dir / f"{stem}_visuals_v8_raw.avi"
        else:
            raw_visual_path = out_dir / f"{stem}_visuals_v8.avi"
        writer = open_video_writer(raw_visual_path, fps=fps, width=width, height=height)

    lock = TargetLock(max_misses=args.max_misses)
    smoother = PoseSmoother(
        conf_thr=max(float(args.joint_conf_thr), args.conf * 1.10),
        median_window=3,
        max_predict_gap=1,
        reset_after_gap=6,
    )

    records: list[dict[str, float | int | bool | None]] = []

    prev_pose: np.ndarray | None = None
    prev_trust: np.ndarray | None = None
    prev_proc_frame_idx: int | None = None

    move_ema = float("nan")
    moving_state = False
    on_counter = 0
    off_counter = 0

    frame_idx = -1
    processed = 0
    no_pose_streak = 0

    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        if args.analysis_stride > 1 and (frame_idx % args.analysis_stride != 0):
            if writer is not None:
                writer.write(frame)
            continue

        processed += 1
        dt = (1.0 / fps) if prev_proc_frame_idx is None else max(1.0 / fps, (frame_idx - prev_proc_frame_idx) / fps)
        prev_proc_frame_idx = frame_idx

        roi = None
        if lock.last_bbox is not None and (processed % max(1, int(args.full_refresh)) != 0):
            roi = lock.roi_box(frame_w=width, frame_h=height, pad=args.roi_pad)

        candidates = iter_predict(model, frame, args=args, roi=roi)
        if not candidates and roi is not None:
            candidates = iter_predict(model, frame, args=args, roi=None)

        candidates = [
            c
            for c in candidates
            if is_plausible_candidate(
                c,
                frame_h=height,
                min_det_conf=args.lock_min_det_conf,
                min_pose_quality=args.lock_min_pose_quality,
                min_torso_px=min_torso_px,
                joint_conf_thr=args.joint_conf_thr,
            )
        ]

        chosen_idx, track_id = lock.select(candidates, frame.shape)

        if chosen_idx is not None:
            c = candidates[chosen_idx]
            pose, trust, joint_age = smoother.step(c.keypoints_xy, c.keypoints_conf, dt=dt)
            pose_quality = c.pose_quality
            det_conf = c.det_conf
            bbox_xyxy = c.bbox_xyxy
        else:
            pose, trust, joint_age = smoother.step(None, None, dt=dt)
            pose_quality = 0.0
            det_conf = float("nan")
            bbox_xyxy = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32)

        torso_px = estimate_torso_px(pose)
        has_reliable_pose = (
            chosen_idx is not None
            and pose_quality >= args.lock_min_pose_quality
            and np.isfinite(torso_px)
            and torso_px >= min_torso_px
            and int(np.sum((trust > 0.35) & (joint_age <= 1))) >= 6
        )
        if has_reliable_pose:
            no_pose_streak = 0
            move_raw = movement_score(prev_pose, pose, prev_trust, trust, torso_px, dt=dt)
            if not np.isfinite(move_raw):
                move_raw = 0.0
            move_raw = float(np.clip(move_raw, 0.0, 0.35))
            draw_pose = pose
            draw_trust = trust
            draw_age = joint_age
        else:
            no_pose_streak += 1
            move_raw = 0.0
            draw_pose = np.full((17, 2), np.nan, dtype=np.float32)
            draw_trust = np.zeros(17, dtype=np.float32)
            draw_age = np.full(17, 99, dtype=np.int32)
            if no_pose_streak >= 6:
                smoother.reset()
                pose = np.full((17, 2), np.nan, dtype=np.float32)
                trust = np.zeros(17, dtype=np.float32)
                joint_age = np.full(17, 99, dtype=np.int32)
                prev_pose = None
                prev_trust = None

        if not np.isfinite(move_ema):
            move_ema = move_raw
        else:
            move_ema = float(args.ema_alpha * move_raw + (1.0 - args.ema_alpha) * move_ema)

        if moving_state:
            if move_ema < args.move_off:
                off_counter += 1
                on_counter = 0
                if off_counter >= args.state_hold_frames:
                    moving_state = False
                    off_counter = 0
            else:
                off_counter = 0
        else:
            if move_ema > args.move_on:
                on_counter += 1
                off_counter = 0
                if on_counter >= args.state_hold_frames:
                    moving_state = True
                    on_counter = 0
            else:
                on_counter = 0

        root_points = draw_pose[[5, 6, 11, 12], :]
        root_valid = np.isfinite(root_points).all(axis=1)
        if np.any(root_valid):
            root_xy = root_points[root_valid].mean(axis=0)
        else:
            root_xy = np.array([np.nan, np.nan], dtype=np.float32)
        valid_joint_count = int(np.sum((draw_trust > 0.25) & (draw_age <= 1)))

        records.append(
            {
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps,
                "dt_sec": dt,
                "track_id": track_id,
                "det_conf": det_conf,
                "pose_quality": pose_quality,
                "movement_raw": move_raw,
                "movement_ema": move_ema,
                "is_moving": moving_state,
                "torso_px": torso_px,
                "valid_joint_count": valid_joint_count,
                "root_x": float(root_xy[0]) if np.isfinite(root_xy[0]) else np.nan,
                "root_y": float(root_xy[1]) if np.isfinite(root_xy[1]) else np.nan,
                "bbox_x1": float(bbox_xyxy[0]),
                "bbox_y1": float(bbox_xyxy[1]),
                "bbox_x2": float(bbox_xyxy[2]),
                "bbox_y2": float(bbox_xyxy[3]),
                "id_switches": lock.id_switches,
            }
        )

        if writer is not None:
            vis = draw_pose_overlay(
                frame=frame,
                pose_xy=draw_pose,
                trust=draw_trust,
                joint_age=draw_age,
                moving=moving_state,
                move_raw=move_raw,
                move_ema=move_ema,
                pose_quality=pose_quality,
                track_id=track_id,
                id_switches=lock.id_switches,
                frame_idx=frame_idx,
                t_sec=frame_idx / fps,
            )
            writer.write(vis)

        if has_reliable_pose:
            prev_pose = pose.copy()
            prev_trust = trust.copy()
        elif chosen_idx is None:
            prev_pose = None
            prev_trust = None

        if processed % 120 == 0:
            elapsed = time.time() - t0
            fps_proc = processed / max(1e-6, elapsed)
            print(f"processed={processed:5d} frame={frame_idx:6d} move_ema={move_ema:.4f} proc_fps={fps_proc:.2f}")

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame.from_records(records)

    ts_csv = out_dir / f"{stem}_timeseries_v8.csv"
    df.to_csv(ts_csv, index=False)

    pause_count, longest_pause = summarize_pauses(df, min_pause_sec=args.min_pause_sec)

    summary = {
        "video": str(video_path),
        "model": model_name,
        "quality": args.quality,
        "frames_total": total_frames,
        "frames_processed": int(len(df)),
        "duration_sec": float(total_frames / fps) if fps > 0 else float("nan"),
        "fps_input": fps,
        "mean_pose_quality": float(df["pose_quality"].mean()) if not df.empty else float("nan"),
        "mean_movement_raw": float(df["movement_raw"].mean()) if not df.empty else float("nan"),
        "mean_movement_ema": float(df["movement_ema"].mean()) if not df.empty else float("nan"),
        "moving_fraction": float(df["is_moving"].mean()) if not df.empty else float("nan"),
        "pause_count": int(pause_count),
        "longest_pause_sec": float(longest_pause),
        "id_switches": int(lock.id_switches),
        "runtime_sec": float(time.time() - t0),
    }

    summary_csv = out_dir / f"{stem}_summary_v8.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    plot_paths = save_plots(df, out_dir=out_dir, stem=stem)

    if args.visuals == "h264" and raw_visual_path is not None:
        final_visual_path = out_dir / f"{stem}_visuals_v8.mp4"
        ok = ffmpeg_h264(raw_visual_path, final_visual_path)
        if ok:
            if not args.keep_raw_avi:
                raw_visual_path.unlink(missing_ok=True)
        else:
            final_visual_path = raw_visual_path
    elif raw_visual_path is not None:
        final_visual_path = raw_visual_path

    print("\n=== outputs ===")
    print(f"timeseries   : {ts_csv}")
    print(f"summary      : {summary_csv}")
    if final_visual_path is not None:
        print(f"visuals      : {final_visual_path}")
    for p in plot_paths:
        print(f"plot         : {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
