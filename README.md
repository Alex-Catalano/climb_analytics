# Climb Analytics

Video-based climbing analysis plus a lightweight session-tracking API.

This repo currently has two main parts:

- A **FastAPI app** for logging climbing sessions and attempts (`/Users/alex/Developer/climb_analytics/climb_analytics`).
- A set of **video analysis scripts** (YOLOv8 and MoveNet) in the repository root.

## What This Project Does

- Tracks climbing sessions, attempts, and rest timing in a local SQLite DB.
- Runs pose-based movement analysis on climbing videos.
- Exports CSV summaries and optional telemetry videos for review.

## Repository Layout

- `/Users/alex/Developer/climb_analytics/climb_analytics/main.py`: FastAPI entrypoint.
- `/Users/alex/Developer/climb_analytics/climb_analytics/models.py`: SQLAlchemy models.
- `/Users/alex/Developer/climb_analytics/climb_analytics/database.py`: DB setup.
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_precision_v8.py`: main accuracy-focused YOLO analyzer (recommended).
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_consolidated_v7.py`: prior consolidated analyzer (legacy reference).
- `/Users/alex/Developer/climb_analytics/tests/fixtures/videos/climb_test_video.mp4`: tracked test fixture video.
- `/Users/alex/Developer/climb_analytics/configs/trackers/bytetrack_climb.yaml`: tracker tuning used by v7.

## Analysis Script Catalog

- `/Users/alex/Developer/climb_analytics/analyze_movenet_pause_summary_stabilized.py`: MoveNet baseline with stabilization and summary-only pause metrics.
- `/Users/alex/Developer/climb_analytics/analyze_movenet_pause_visuals.py`: MoveNet pause analysis with CSV + movement plot output.
- `/Users/alex/Developer/climb_analytics/analyze_movenet_jump_telemetry_v2.py`: MoveNet jump-focused pipeline with crop tracking, telemetry output, and proxy flow.
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_precision_v8.py`: YOLO v8 rewrite for higher pose stability and annotated output video (current recommended script).
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_active_decay_v3.py`: YOLOv8 active-climb segmentation and decay windows (v3 generation).
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_metrics_decay_v4.py`: YOLOv8 robust movement + climber metrics + decay windows (stable legacy baseline).
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_autocrop_oneeuro_v5.py`: YOLOv8 v5 with saved auto-crop and One Euro keypoint smoothing.
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_tracking_sanity_v6.py`: YOLOv8 v6 with tracker-assisted pose and skeletal sanity gating.
- `/Users/alex/Developer/climb_analytics/analyze_yolov8_climb_consolidated_v7.py`: YOLOv8 v7 with track-id target locking and v6 safety logic (legacy fallback).

## Quick Start

1. Activate your virtual environment (if you use one):

```bash
source venv/bin/activate
```

2. Discover available commands:

```bash
make
```

3. Run the API:

```bash
make api
```

4. In another terminal, check health:

```bash
make api-health
```

5. Run YOLO analysis (recommended):

```bash
make analyze-v8 VIDEO=climbing_clip_1_proxy.mp4 OUT=outputs_v8 QUALITY=max
```

## Makefile Commands

`make` defaults to `help`.

- `make help`
  - Shows available targets, variable defaults, and examples.
- `make show-config`
  - Prints resolved runtime variables.
- `make api`
  - Starts FastAPI with Uvicorn.
- `make api-health`
  - Calls local `/health`.
- `make analyze-v4`
  - Runs `analyze_yolov8_climb_metrics_decay_v4.py` (no telemetry output video).
- `make analyze-v4-telemetry`
  - Runs v4 analysis and emits telemetry AVI/H264 outputs.
- `make analyze-v8`
  - Runs `analyze_yolov8_climb_precision_v8.py` (accuracy-focused, annotated H.264 video).
- `make analyze-v8-data`
  - Runs v8 data/plots only (no output video).
- `make analyze-v8-fast`
  - Runs v8 fast preset for quick iteration.
- `make analyze-v8-max`
  - Runs v8 max preset (slowest, highest default accuracy).
- `make analyze-v7`
  - Runs `analyze_yolov8_climb_consolidated_v7.py` (legacy path) and writes annotated H.264 video.
- `make analyze-v7-data`
  - Runs v7 without annotated video (data + plots only).
- `make analyze-v7-max`
  - Runs v7 at max-quality preset (`ANALYSIS_STRIDE=1`, larger inference size).
- `make analyze-v7-telemetry`
  - Runs v7 and emits telemetry AVI/H264 outputs using explicit telemetry flags.
- `make test-video`
  - Copies `/Users/alex/Developer/climb_analytics/climb_analytics/clip10.mp4` into the tracked fixture path.
- `make clean-outputs`
  - Removes generated output folders.

## Makefile Variables

You can override variables at runtime, for example:

```bash
make analyze-v8 VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_test QUALITY=max
```

CLI shortcut in v8 for visuals:

```bash
python analyze_yolov8_climb_precision_v8.py climbing_clip_1_proxy.mp4 --out outputs_v8 --quality max --visuals h264
```

Supported variables:

- `PY`: Python executable (defaults to `venv/bin/python` when available, else `python3`).
- `VIDEO`: input video path for analysis commands.
- `OUT`: output directory for generated files.
- `MODEL`: YOLO model path (`yolov8n-pose.pt`, `yolov8s-pose.pt`, etc.).
- `QUALITY`: quality preset (`fast`, `balanced`, `max`) used by v8 and v7 targets.
- `HOST`: API bind host.
- `PORT`: API bind port.
- `FIXTURE_DIR`: directory for the tracked test fixture.
- `FIXTURE_VIDEO`: full path for fixture output video.
- `TRACKER`: tracker config path used by v7.

## Notes

- Large `.mp4/.MOV/.avi` files are ignored globally.
- The fixture exception `!tests/fixtures/videos/*.mp4` allows a deliberate tracked test video.
- Model files (`*.pt`, `*.tflite`) are currently tracked by design.
