import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Tunables (you will tune these)
# -----------------------------

# Movement threshold: if "movement" is above this, we label it as moving.
# If below, we treat it as "resting / pausing".
MOVEMENT_THRESHOLD = 0.020

# How long a pause has to last (seconds) before we count it as a real pause segment
MIN_PAUSE_SECONDS = 0.35

# Speed control: analyze every Nth frame.
# 3 on a ~30fps video => ~10 fps analysis (way faster and still good enough for pause detection)
FRAME_STRIDE = 3

# Stabilization feature tracking settings (lower = faster; higher = potentially better stabilization)
MAX_CORNERS = 120
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 15

# MoveNet model path (expects the .tflite file sitting next to this script)
MODEL_PATH = str(Path(__file__).resolve().parent / "movenet_singlepose_lightning.tflite")

# Pose confidence threshold (we debugged your video and the confidences were ~0.01-0.09)
# So this MUST be low, otherwise you'd get pose_detect_rate ~ 0.
CONF_THRESHOLD = 0.03

# Require at least this many keypoints "confident" to say pose exists
MIN_GOOD_KPTS = 5

# Require at least this many overlapping keypoints between frame t and t-1 to compute movement
MIN_OVERLAP_KPTS = 6

# Debug toggle: set True if you want to print keypoint stats on the first frame
DEBUG = False


# -----------------------------------
# Video stabilization (optional helper)
# -----------------------------------
def stabilize_frames(cap: cv2.VideoCapture):
    """
    Yields "stabilized" frames from a VideoCapture.

    How it works:
    - Track feature points from prev frame to current using optical flow (Lucas-Kanade).
    - Estimate an affine transform (translation/rotation/scale) between frames.
    - Accumulate transforms, then invert to keep the camera "steady".

    This helps if your phone/camera moves while filming.
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

        # Find trackable points in the previous frame
        p0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=QUALITY_LEVEL,
            minDistance=MIN_DISTANCE,
            blockSize=7,
        )

        if p0 is None:
            yield frame
            prev_gray = gray
            continue

        # Track those points into the new frame
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
        if p1 is None:
            yield frame
            prev_gray = gray
            continue

        good0 = p0[st.flatten() == 1]
        good1 = p1[st.flatten() == 1]

        # If we don't have enough points, skip stabilization this frame
        if len(good0) < 10:
            yield frame
            prev_gray = gray
            continue

        # Estimate frame-to-frame transform (affine, no perspective)
        M, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC)
        if M is None:
            yield frame
            prev_gray = gray
            continue

        # Convert 2x3 affine into a 3x3 matrix so we can chain multiplications
        M_3 = np.array(
            [[M[0, 0], M[0, 1], M[0, 2]],
             [M[1, 0], M[1, 1], M[1, 2]],
             [0,       0,       1]],
            dtype=np.float32
        )

        # Accumulate transforms over time
        M_cum = M_3 @ M_cum

        # Apply inverse transform to "undo" the camera motion
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
    We try tflite_runtime first (lightweight), otherwise fall back to tensorflow.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}\n"
            f"Make sure movenet_singlepose_lightning.tflite is next to this script."
        )

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


def run_movenet(interpreter, input_details, output_details, frame_bgr):
    """
    Runs MoveNet on one frame and returns keypoints.

    Output format: (17, 3) for singlepose
      - [y, x, confidence] in normalized coordinates relative to the model input.
    """
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)

    # This model expects uint8 input (we fixed a dtype mismatch earlier)
    inp = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]["index"], inp)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]["index"])
    kpts = out[0, 0, :, :]  # shape: [17, 3]
    return kpts


# -----------------------
# Main analysis pipeline
# -----------------------
def analyze_video(video_path: str) -> dict:
    """
    Goal:
    - Detect pose keypoints over time (MoveNet)
    - Convert pose change into a simple "movement score"
    - Use movement score to classify moving vs pausing
    - Summarize: total pause time, number of pauses, etc.

    Note: This is *rough* movement, not perfect "climbing progress".
    It’s a starting signal for fatigue / rest patterns.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    interpreter, input_details, output_details = load_interpreter()

    prev_pts = None       # previous frame (17,2) keypoints
    prev_good = None      # previous frame (17,) confidence mask

    scores = []           # movement score per processed frame
    times = []            # timestamp (seconds) per processed frame
    has_pose = []         # whether we consider pose usable for this frame

    frame_idx = 0         # counts ALL frames read (even skipped)

    for frame in stabilize_frames(cap):
        # Skip frames for speed (but still advance frame_idx)
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        # True timestamp in the original video
        t = frame_idx / fps

        # Pose detection
        kpts = run_movenet(interpreter, input_details, output_details, frame)

        if DEBUG and frame_idx == 0:
            conf = kpts[:, 2]
            print("kpts shape:", kpts.shape)
            print("conf min/mean/max:", float(conf.min()), float(conf.mean()), float(conf.max()))
            print("first 5 kpts:", kpts[:5])

        conf = kpts[:, 2]      # (17,)
        pts = kpts[:, :2]      # (17,2) y,x normalized
        good = conf > CONF_THRESHOLD

        # Default movement score. 0.0 means "no measurable motion".
        # (Using 0 instead of NaN makes stats and thresholds easier.)
        score = 0.0

        if good.sum() >= MIN_GOOD_KPTS:
            if prev_pts is not None and prev_good is not None:
                # Compare only joints that are confident in both consecutive frames
                both = good & prev_good
                if both.sum() >= MIN_OVERLAP_KPTS:
                    diffs = pts[both] - prev_pts[both]
                    score = float(np.linalg.norm(diffs, axis=1).mean())
                else:
                    score = 0.0  # not enough overlap; treat as "unknown but small"

            # Update "previous" for next frame
            prev_pts = pts
            prev_good = good
            has_pose.append(True)
        else:
            # Pose is too weak this frame, reset prev so we don't compare garbage
            prev_pts = None
            prev_good = None
            has_pose.append(False)
            score = 0.0

        scores.append(score)
        times.append(t)

        frame_idx += 1

    cap.release()

    if len(times) == 0:
        return {"video": Path(video_path).name, "error": "empty video or all frames skipped"}

    # Build a dataframe for easy processing
    df = pd.DataFrame({"t": times, "movement": scores, "has_pose": has_pose})

    # "Moving" vs "not moving"
    df["is_moving"] = df["movement"] >= MOVEMENT_THRESHOLD

    # Duration = last timestamp (because times are real video time)
    duration = float(df["t"].iloc[-1])

    # Find pause segments: consecutive intervals of "not moving" lasting >= MIN_PAUSE_SECONDS
    pause_segments = []
    in_pause = False
    start = 0.0

    for _, row in df.iterrows():
        if (not row["is_moving"]) and (not in_pause):
            in_pause = True
            start = row["t"]
        if row["is_moving"] and in_pause:
            end = row["t"]
            in_pause = False
            if end - start >= MIN_PAUSE_SECONDS:
                pause_segments.append((start, end))

    # If we ended while still paused, close it out
    if in_pause:
        end = duration
        if end - start >= MIN_PAUSE_SECONDS:
            pause_segments.append((start, end))

    pause_time = sum(e - s for s, e in pause_segments)
    pause_count = len(pause_segments)
    move_time = max(0.0, duration - pause_time)

    avg_movement = float(df["movement"].mean())
    total_movement = float(df["movement"].sum())
    movement_rate = float(total_movement / duration) if duration > 0 else 0.0

    return {
        "video": Path(video_path).name,
        "duration_s": round(duration, 2),
        "pause_time_s": round(pause_time, 2),
        "pause_count": pause_count,
        "move_time_s": round(move_time, 2),
        "avg_movement": round(avg_movement, 6),
        "movement_rate": round(movement_rate, 6),
        "pose_detect_rate": round(float(df["has_pose"].mean()), 3),
        "used_fps": round(float(fps / FRAME_STRIDE), 2),  # approx analysis FPS
        "conf_threshold": CONF_THRESHOLD,
        "movement_threshold": MOVEMENT_THRESHOLD,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_movenet_pause_summary_stabilized.py path/to/video.mp4")
        raise SystemExit(1)

    print(analyze_video(sys.argv[1]))