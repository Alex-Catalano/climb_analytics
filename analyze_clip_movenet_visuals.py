import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# Optional plotting (only needed for visualization)
import matplotlib.pyplot as plt


# -----------------------------
# Tunables (you will tune these)
# -----------------------------

# Movement threshold:
# If "movement" >= this -> moving
# If below -> pausing (micro-pause) *inside the attempt*
MOVEMENT_THRESHOLD = 0.035  # was 0.020 (too sensitive)

# How long a pause must last to count as a pause segment.
# This is the biggest lever to stop "fake rests" during one attempt.
MIN_PAUSE_SECONDS = 1.25  # was 0.35 (counts every hand readjust as "pause")

# Speed control: analyze every Nth frame for performance.
# 3 on a ~30fps video => ~10 fps analysis (fast and usually enough)
FRAME_STRIDE = 3

# Stabilization feature tracking settings (lower = faster; higher can stabilize better)
MAX_CORNERS = 120
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 15

# MoveNet model path: .tflite file must be next to this script
MODEL_PATH = str(Path(__file__).resolve().parent / "movenet_singlepose_lightning.tflite")

# Pose confidence threshold:
# Your clip had low confidences (~0.01–0.09), so this must be low.
CONF_THRESHOLD = 0.03

# Require at least this many confident keypoints to say "pose exists"
MIN_GOOD_KPTS = 5

# Require at least this many overlapping joints between consecutive frames to compute movement
MIN_OVERLAP_KPTS = 6

# If two pauses are separated by a tiny moving gap, merge them (prevents fragmentation)
MERGE_GAP_SECONDS = 0.35

# Debug: prints keypoint stats on the first processed frame
DEBUG = False


# -----------------------------------
# Video stabilization (optional helper)
# -----------------------------------
def stabilize_frames(cap: cv2.VideoCapture):
    """
    Yields "stabilized" frames from a VideoCapture.

    Why:
    - If your phone/camera shifts a bit, pose keypoints jump even if you don't move.
    - Stabilization reduces that camera-induced motion.

    How:
    - Track feature points frame->frame (optical flow).
    - Estimate an affine transform (translation/rotation/scale).
    - Accumulate transforms and invert them so the camera looks steady.
    """
    ok, prev = cap.read()
    if not ok:
        return

    h, w = prev.shape[:2]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    # cumulative transform (identity to start)
    M_cum = np.eye(3, dtype=np.float32)

    yield prev

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=7,
        )

        # If no features, just give original frame
        if p0 is None:
            yield frame
            prev_gray = gray
            continue

        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        if p1 is None:
            yield frame
            prev_gray = gray
            continue

        good0 = p0[st.flatten() == 1]
        good1 = p1[st.flatten() == 1]

        # If too few points, skip stabilization this frame
        if len(good0) < 10:
            yield frame
            prev_gray = gray
            continue

        M, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC)
        if M is None:
            yield frame
            prev_gray = gray
            continue

        # Convert 2x3 affine -> 3x3 so we can multiply cumulatively
        M_3 = np.array(
            [[M[0, 0], M[0, 1], M[0, 2]],
             [M[1, 0], M[1, 1], M[1, 2]],
             [0,       0,       1]],
            dtype=np.float32
        )

        # Accumulate and invert to "undo" camera motion
        M_cum = M_3 @ M_cum
        M_inv = np.linalg.inv(M_cum)

        stabilized = cv2.warpAffine(
            frame,
            M_inv[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        yield stabilized
        prev_gray = gray


# -----------------------
# MoveNet model utilities
# -----------------------
def load_interpreter():
    """
    Loads the TFLite interpreter.
    Tries tflite_runtime first (lightweight), otherwise falls back to tensorflow.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_path}\n"
            f"Make sure movenet_singlepose_lightning.tflite is next to this script."
        )

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def run_movenet(interpreter, input_details, output_details, frame_bgr):
    """
    Runs MoveNet on one frame and returns keypoints.

    Output format: (17, 3)
      - [y, x, confidence] normalized relative to the model input.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)

    # Your TFHub tflite expects uint8 input (we fixed the dtype mismatch earlier)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    return out[0, 0, :, :]  # (17,3)


# -----------------------
# Pause segmentation helpers
# -----------------------
def _segments_from_boolean(times, is_paused):
    """
    Convert a boolean "paused" signal into time segments (start,end).
    """
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

    # close final pause if it ends at the end
    if in_pause and start is not None:
        segs.append((float(start), float(times[-1])))

    return segs


def _filter_and_merge_segments(segs, min_len_s, merge_gap_s):
    """
    - drop short segments (< min_len_s)
    - merge segments separated by small gaps (<= merge_gap_s)
    """
    # drop short
    segs = [(s, e) for (s, e) in segs if (e - s) >= min_len_s]
    if not segs:
        return []

    # sort then merge
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
# Main analysis pipeline
# -----------------------
def analyze_video(video_path: str):
    """
    What this produces:
    - A per-time movement score derived from pose keypoints.
    - A moving vs pausing (micro-pause) classification.
    - Pause segments + summary metrics.

    Important interpretation:
    - Within a single climb attempt, you WILL micro-pause on holds.
    - So "pause_count" is *micro-pauses*, not "rests between attempts".
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    interpreter, input_details, output_details = load_interpreter()

    prev_pts = None       # previous keypoints (17,2)
    prev_good = None      # previous confidence mask (17,)

    movement_scores = []  # movement per processed frame
    times = []            # timestamps (seconds) per processed frame
    has_pose = []         # whether pose was usable

    frame_idx = 0  # counts frames read (including skipped)

    for frame in stabilize_frames(cap):
        # Skip frames for speed
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        t = frame_idx / fps
        kpts = run_movenet(interpreter, input_details, output_details, frame)

        if DEBUG and len(times) == 0:
            conf0 = kpts[:, 2]
            print("kpts shape:", kpts.shape)
            print("conf min/mean/max:", float(conf0.min()), float(conf0.mean()), float(conf0.max()))
            print("first 5 kpts:", kpts[:5])

        conf = kpts[:, 2]         # (17,)
        pts = kpts[:, :2]         # (17,2) y,x
        good = conf > CONF_THRESHOLD

        # Default movement score for this frame
        score = 0.0

        if good.sum() >= MIN_GOOD_KPTS:
            if prev_pts is not None and prev_good is not None:
                both = good & prev_good
                if both.sum() >= MIN_OVERLAP_KPTS:
                    diffs = pts[both] - prev_pts[both]
                    # average joint displacement magnitude
                    score = float(np.linalg.norm(diffs, axis=1).mean())
                else:
                    # not enough overlap to compute confidently
                    score = 0.0

            prev_pts = pts
            prev_good = good
            has_pose.append(True)
        else:
            # Pose weak this frame -> reset so we don't compare garbage
            prev_pts = None
            prev_good = None
            has_pose.append(False)
            score = 0.0

        times.append(float(t))
        movement_scores.append(float(score))

        frame_idx += 1

    cap.release()

    if len(times) == 0:
        raise RuntimeError("No frames processed (video empty or all frames skipped).")

    # Build dataframe you can export + plot
    df = pd.DataFrame({
        "t": times,
        "movement": movement_scores,
        "has_pose": has_pose,
    })

    # Classification
    df["is_moving"] = df["movement"] >= MOVEMENT_THRESHOLD
    df["is_paused"] = ~df["is_moving"]

    duration = float(df["t"].iloc[-1])

    # Create pause segments from boolean signal and then filter/merge
    raw_pause_segments = _segments_from_boolean(df["t"].values, df["is_paused"].values)
    pause_segments = _filter_and_merge_segments(
        raw_pause_segments,
        min_len_s=MIN_PAUSE_SECONDS,
        merge_gap_s=MERGE_GAP_SECONDS,
    )

    pause_time = float(sum(e - s for s, e in pause_segments))
    pause_count = int(len(pause_segments))
    move_time = float(max(0.0, duration - pause_time))

    # Summary stats
    avg_movement = float(df["movement"].mean())
    total_movement = float(df["movement"].sum())
    movement_rate = float(total_movement / duration) if duration > 0 else 0.0
    pose_detect_rate = float(df["has_pose"].mean())

    summary = {
        "video": Path(video_path).name,
        "duration_s": round(duration, 2),

        # This is micro-pauses *inside* the attempt (not “rest between attempts”)
        "micro_pause_time_s": round(pause_time, 2),
        "micro_pause_count": pause_count,

        "move_time_s": round(move_time, 2),
        "avg_movement": round(avg_movement, 6),
        "movement_rate": round(movement_rate, 6),
        "pose_detect_rate": round(pose_detect_rate, 3),

        "used_fps": round(float(fps / FRAME_STRIDE), 2),
        "conf_threshold": CONF_THRESHOLD,
        "movement_threshold": MOVEMENT_THRESHOLD,
        "min_pause_seconds": MIN_PAUSE_SECONDS,
        "merge_gap_seconds": MERGE_GAP_SECONDS,
    }

    return summary, df, pause_segments


def save_outputs(video_path: str, df: pd.DataFrame, pause_segments, out_dir: Path):
    """
    Writes:
    - CSV with per-time movement values
    - PNG plot of movement over time with pause regions shaded
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    csv_path = out_dir / f"{stem}_movement.csv"
    png_path = out_dir / f"{stem}_movement.png"

    # Save raw numbers
    df.to_csv(csv_path, index=False)

    # Plot: movement vs time, with threshold line and pause shading
    plt.figure()
    plt.plot(df["t"], df["movement"])
    plt.axhline(MOVEMENT_THRESHOLD, linestyle="--")

    # Shade pause segments
    for s, e in pause_segments:
        plt.axvspan(s, e, alpha=0.25)

    plt.xlabel("Time (s)")
    plt.ylabel("Movement score (avg joint delta)")
    plt.title(f"{stem} movement + micro-pauses")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    return csv_path, png_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze a climbing clip using MoveNet + optional stabilization and output CSV/plot."
    )
    parser.add_argument("video", help="Path to video file (mp4/mov)")
    parser.add_argument(
        "--out",
        default="outputs",
        help="Output folder for CSV/PNG (default: ./outputs)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print keypoint confidence stats for first processed frame",
    )
    args = parser.parse_args()

    DEBUG = bool(args.debug)

    summary, df, pause_segments = analyze_video(args.video)
    print(summary)

    csv_path, png_path = save_outputs(args.video, df, pause_segments, Path(args.out))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")