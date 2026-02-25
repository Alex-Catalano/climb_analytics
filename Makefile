.DEFAULT_GOAL := help

PY ?= python3
VENV_PY := venv/bin/python
ifeq ($(wildcard $(VENV_PY)),$(VENV_PY))
PY := $(VENV_PY)
endif

VIDEO ?= climbing_clip_1_proxy.mp4
OUT ?= outputs
MODEL ?= yolov8n-pose.pt
QUALITY ?= max
HOST ?= 127.0.0.1
PORT ?= 8000
FIXTURE_DIR ?= tests/fixtures/videos
FIXTURE_VIDEO ?= $(FIXTURE_DIR)/climb_test_video.mp4
TRACKER ?= configs/trackers/bytetrack_climb.yaml

.PHONY: help show-config api api-health analyze-v4 analyze-v4-telemetry analyze-v7 analyze-v7-data analyze-v7-max analyze-v7-telemetry analyze-v8 analyze-v8-data analyze-v8-fast analyze-v8-max test-video clean-outputs

help: ## Show available commands and defaults.
	@echo "climb_analytics Makefile"
	@echo
	@echo "Usage:"
	@echo "  make <target> [VAR=value]"
	@echo
	@echo "Main targets:"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Variables:"
	@echo "  PY=$(PY)"
	@echo "  VIDEO=$(VIDEO)"
	@echo "  OUT=$(OUT)"
	@echo "  MODEL=$(MODEL)"
	@echo "  QUALITY=$(QUALITY)"
	@echo "  HOST=$(HOST)"
	@echo "  PORT=$(PORT)"
	@echo "  FIXTURE_VIDEO=$(FIXTURE_VIDEO)"
	@echo "  TRACKER=$(TRACKER)"
	@echo
	@echo "Examples:"
	@echo "  make analyze-v4 VIDEO=climbing_clip_1_proxy.mp4 OUT=outputs_demo"
	@echo "  make analyze-v8 VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v8 QUALITY=max"
	@echo "  make analyze-v8-data VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v8_data QUALITY=max"
	@echo "  make analyze-v8-fast VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v8_fast"
	@echo "  make analyze-v7 VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v7 QUALITY=balanced"
	@echo "  make analyze-v7-data VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v7_data QUALITY=balanced"
	@echo "  make analyze-v7-max VIDEO=tests/fixtures/videos/climb_test_video.mp4 OUT=outputs_v7_max"
	@echo "  make analyze-v4-telemetry VIDEO=tests/fixtures/videos/climb_test_video.mp4"
	@echo "  make api HOST=0.0.0.0 PORT=8000"

show-config: ## Print resolved configuration values.
	@echo "PY=$(PY)"
	@echo "VIDEO=$(VIDEO)"
	@echo "OUT=$(OUT)"
	@echo "MODEL=$(MODEL)"
	@echo "QUALITY=$(QUALITY)"
	@echo "HOST=$(HOST)"
	@echo "PORT=$(PORT)"
	@echo "FIXTURE_VIDEO=$(FIXTURE_VIDEO)"
	@echo "TRACKER=$(TRACKER)"

api: ## Run the FastAPI app via Uvicorn.
	$(PY) -m uvicorn climb_analytics.main:app --reload --host $(HOST) --port $(PORT)

api-health: ## Query the local API health endpoint.
	curl -fsS http://$(HOST):$(PORT)/health

analyze-v4: ## Run YOLOv8 v4 analysis (CSV + plots; no telemetry video).
	$(PY) analyze_yolov8_climb_metrics_decay_v4.py $(VIDEO) --out $(OUT) --model $(MODEL)

analyze-v4-telemetry: ## Run YOLOv8 v4 analysis and write telemetry outputs.
	$(PY) analyze_yolov8_climb_metrics_decay_v4.py $(VIDEO) --out $(OUT) --model $(MODEL) --telemetry --telemetry-h264

analyze-v7: ## Run consolidated YOLOv8 v7 and save annotated H.264 visuals.
	$(PY) analyze_yolov8_climb_consolidated_v7.py $(VIDEO) --out $(OUT) --quality $(QUALITY) --tracker $(TRACKER) --visuals h264

analyze-v7-data: ## Run consolidated YOLOv8 v7 data/plots only (no telemetry video).
	$(PY) analyze_yolov8_climb_consolidated_v7.py $(VIDEO) --out $(OUT) --quality $(QUALITY) --tracker $(TRACKER) --visuals none

analyze-v7-max: ## Run v7 with max-quality preset.
	$(PY) analyze_yolov8_climb_consolidated_v7.py $(VIDEO) --out $(OUT) --quality max --tracker $(TRACKER) --visuals h264

analyze-v7-telemetry: ## Run v7 with telemetry AVI/H264 outputs.
	$(PY) analyze_yolov8_climb_consolidated_v7.py $(VIDEO) --out $(OUT) --quality $(QUALITY) --tracker $(TRACKER) --telemetry --telemetry-h264

analyze-v8: ## Run v8 accuracy-focused YOLO analyzer with annotated H.264 visuals (recommended).
	$(PY) analyze_yolov8_climb_precision_v8.py $(VIDEO) --out $(OUT) --quality $(QUALITY) --visuals h264

analyze-v8-data: ## Run v8 data/plots only (no output video).
	$(PY) analyze_yolov8_climb_precision_v8.py $(VIDEO) --out $(OUT) --quality $(QUALITY) --visuals none

analyze-v8-fast: ## Run v8 in fast preset for quick iteration.
	$(PY) analyze_yolov8_climb_precision_v8.py $(VIDEO) --out $(OUT) --quality fast --visuals h264

analyze-v8-max: ## Run v8 in max preset (highest accuracy, slowest).
	$(PY) analyze_yolov8_climb_precision_v8.py $(VIDEO) --out $(OUT) --quality max --visuals h264

test-video: ## Create/update tracked test video fixture from climb_analytics/clip10.mp4.
	mkdir -p $(FIXTURE_DIR)
	cp -f climb_analytics/clip10.mp4 $(FIXTURE_VIDEO)
	@echo "Fixture ready: $(FIXTURE_VIDEO)"

clean-outputs: ## Remove generated output folders.
	rm -rf outputs_demo_v4
	rm -rf outputs
