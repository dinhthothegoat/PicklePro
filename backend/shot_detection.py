"""Whole-match shot detection helpers.

The detector is designed around a pluggable model path. If an Ultralytics YOLO
ball model is available locally, it can be used for ball detections. Otherwise
the app falls back to a lightweight motion/color tracker so the prototype still
produces a full-match timeline without downloading large weights.
"""

from __future__ import annotations

from collections import deque
import os
import threading
from typing import Callable

import numpy as np


ProgressCallback = Callable[[dict], None]
_YOLO_MODEL = None
_YOLO_MODEL_ERROR = None
_YOLO_LOCK = threading.Lock()


class WholeMatchShotTracker:
    """Streaming shot tracker that does not keep full frames in memory."""

    def __init__(self, width: int, height: int, fps: float, duration_seconds: float):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.previous_frame = None
        self.previous_track = None
        self.tracks = []
        self.frames_seen = 0
        self.yolo_model = load_yolo_model()
        self.detector_name = "YOLO ball detector" if self.yolo_model else "motion-color fallback tracker"

    def add_frame(self, frame_rgb: np.ndarray, frame_index: int, timestamp: float) -> None:
        self.frames_seen += 1
        candidate = detect_ball_candidate(frame_rgb, self.previous_frame, yolo_model=self.yolo_model)
        self.previous_frame = frame_rgb
        if not candidate:
            return

        x = candidate["x"] * self.width
        y = candidate["y"] * self.height
        velocity = 0.0
        if self.previous_track:
            dt = max(0.001, timestamp - self.previous_track["timestamp"])
            velocity = (((x - self.previous_track["x"]) ** 2 + (y - self.previous_track["y"]) ** 2) ** 0.5) / dt
        track = {
            "timestamp": round(timestamp, 2),
            "frame_index": frame_index,
            "x": round(x, 1),
            "y": round(y, 1),
            "confidence": round(candidate["confidence"], 2),
            "velocity": round(velocity, 1),
        }
        self.tracks.append(track)
        self.previous_track = track

    def finish(self) -> dict:
        shot_events = infer_shot_events(self.tracks, self.duration_seconds)
        rally_windows = build_rally_windows(shot_events, self.duration_seconds)
        shot_mix = summarize_shot_mix(shot_events, self.width, self.height)
        return {
            "detector": self.detector_name,
            "model_reference": model_reference(self.yolo_model),
            "model_error": _YOLO_MODEL_ERROR,
            "tracks": self.tracks[-240:],
            "track_points": len(self.tracks),
            "events": shot_events,
            "rally_windows": rally_windows,
            "shot_mix": shot_mix,
            "coverage": {
                "duration_seconds": round(self.duration_seconds, 1),
                "sampled_frames": self.frames_seen,
                "tracked_frames": len(self.tracks),
                "track_rate": round(len(self.tracks) / max(1, self.frames_seen), 2),
            },
        }


def analyze_shots_from_samples(
    sampled_frames: list[dict],
    width: int,
    height: int,
    fps: float,
    duration_seconds: float,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    tracks = []
    previous = None
    yolo_model = load_yolo_model()
    detector_name = "YOLO ball detector" if yolo_model else "motion-color fallback tracker"

    for index, sample in enumerate(sampled_frames):
        candidate = detect_ball_candidate(
            sample.get("small_rgb"),
            sample.get("previous_small_rgb"),
            yolo_model=yolo_model,
        )
        if candidate:
            x = candidate["x"] * width
            y = candidate["y"] * height
            timestamp = sample["timestamp"]
            velocity = 0.0
            if previous:
                dt = max(0.001, timestamp - previous["timestamp"])
                velocity = (((x - previous["x"]) ** 2 + (y - previous["y"]) ** 2) ** 0.5) / dt
            track = {
                "timestamp": round(timestamp, 2),
                "frame_index": sample["frame_index"],
                "x": round(x, 1),
                "y": round(y, 1),
                "confidence": round(candidate["confidence"], 2),
                "velocity": round(velocity, 1),
            }
            tracks.append(track)
            previous = track

        if progress_callback and index and index % 45 == 0:
            progress_callback(
                {
                    "percent": min(82, 55 + int(index / max(1, len(sampled_frames)) * 27)),
                    "phase": "Detecting shots",
                    "message": f"Tracked ball candidates in {index} sampled frames.",
                    "frames_analyzed": index,
                }
            )

    shot_events = infer_shot_events(tracks, duration_seconds)
    rally_windows = build_rally_windows(shot_events, duration_seconds)
    shot_mix = summarize_shot_mix(shot_events, width, height)

    return {
        "detector": detector_name,
        "model_reference": model_reference(yolo_model),
        "model_error": _YOLO_MODEL_ERROR,
        "tracks": tracks[-240:],
        "track_points": len(tracks),
        "events": shot_events,
        "rally_windows": rally_windows,
        "shot_mix": shot_mix,
        "coverage": {
            "duration_seconds": round(duration_seconds, 1),
            "sampled_frames": len(sampled_frames),
            "tracked_frames": len(tracks),
            "track_rate": round(len(tracks) / max(1, len(sampled_frames)), 2),
        },
    }


def load_yolo_model():
    global _YOLO_MODEL, _YOLO_MODEL_ERROR
    if _YOLO_MODEL is not None or _YOLO_MODEL_ERROR is not None:
        return _YOLO_MODEL
    with _YOLO_LOCK:
        if _YOLO_MODEL is not None or _YOLO_MODEL_ERROR is not None:
            return _YOLO_MODEL

        model_path = os.getenv("SHOT_DETECTOR_MODEL_PATH", "").strip()
        if not model_path:
            _YOLO_MODEL_ERROR = "No SHOT_DETECTOR_MODEL_PATH configured; using fallback tracker."
            return None

        try:
            from ultralytics import YOLO  # type: ignore[import]

            _YOLO_MODEL = YOLO(model_path)
            return _YOLO_MODEL
        except Exception as exc:
            _YOLO_MODEL_ERROR = f"Could not load YOLO model from SHOT_DETECTOR_MODEL_PATH: {exc}"
            return None


def model_reference(yolo_model) -> str:
    if yolo_model:
        return "Using local YOLO ball-detector weights configured by SHOT_DETECTOR_MODEL_PATH."
    return "Compatible with TrackNet/YOLO pickleball detectors; fallback used when no local weights are configured."


def detect_ball_candidate(rgb: np.ndarray | None, previous_rgb: np.ndarray | None, yolo_model=None):
    if rgb is None:
        return None
    if rgb.dtype != np.float64 and rgb.dtype != np.float32:
        rgb = rgb.astype(float) / 255.0
    if previous_rgb is not None and previous_rgb.dtype != np.float64 and previous_rgb.dtype != np.float32:
        previous_rgb = previous_rgb.astype(float) / 255.0
    if yolo_model:
        yolo_candidate = detect_with_yolo(rgb, yolo_model)
        if yolo_candidate:
            return yolo_candidate

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    bright_ball = (green > 0.58) & (red > 0.45) & (blue < 0.42) & ((green - blue) > 0.18)

    motion_mask = np.zeros_like(green, dtype=bool)
    if previous_rgb is not None:
        diff = np.abs(rgb.mean(axis=2) - previous_rgb.mean(axis=2))
        threshold = max(0.08, float(diff.mean() + diff.std() * 1.8))
        motion_mask = diff > threshold

    candidate_mask = bright_ball | (motion_mask & (green > blue * 1.05))
    if candidate_mask.mean() < 0.00015:
        return None

    y_idx, x_idx = np.where(candidate_mask)
    if len(x_idx) == 0:
        return None

    weights = green[y_idx, x_idx] + red[y_idx, x_idx]
    x = float(np.average(x_idx, weights=weights)) / max(1, rgb.shape[1] - 1)
    y = float(np.average(y_idx, weights=weights)) / max(1, rgb.shape[0] - 1)
    confidence = min(1.0, 0.28 + candidate_mask.mean() * 90 + float(weights.mean()) * 0.28)
    return {"x": x, "y": y, "confidence": confidence}


def detect_with_yolo(rgb: np.ndarray, yolo_model):
    frame = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    results = yolo_model.predict(frame, verbose=False)
    if not results:
        return None
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    for box in boxes:
        confidence = float(box.conf[0])
        if best is None or confidence > best[0]:
            best = (confidence, box)
    if best is None:
        return None

    confidence, box = best
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
    return {
        "x": ((x1 + x2) / 2) / max(1, frame.shape[1]),
        "y": ((y1 + y2) / 2) / max(1, frame.shape[0]),
        "confidence": min(1.0, confidence),
    }


def infer_shot_events(tracks: list[dict], duration_seconds: float) -> list[dict]:
    if len(tracks) < 4:
        return []

    events = []
    recent = deque(maxlen=5)
    last_event_time = -10.0

    for point in tracks:
        recent.append(point)
        if len(recent) < 4:
            continue

        velocities = [item["velocity"] for item in recent if item["velocity"]]
        if not velocities:
            continue
        velocity = velocities[-1]
        avg_velocity = sum(velocities) / len(velocities)
        direction_change = changed_direction(list(recent))
        enough_gap = point["timestamp"] - last_event_time >= 1.2
        if enough_gap and (velocity > avg_velocity * 1.45 or direction_change):
            events.append(
                {
                    "time": round(point["timestamp"], 1),
                    "type": classify_shot(point),
                    "court_zone": classify_zone(point),
                    "speed_index": round(min(1.0, velocity / 1200), 2),
                    "confidence": point["confidence"],
                }
            )
            last_event_time = point["timestamp"]

    if not events and tracks:
        interval = max(8, int(duration_seconds // 10) or 8)
        events = [
            {
                "time": round(point["timestamp"], 1),
                "type": classify_shot(point),
                "court_zone": classify_zone(point),
                "speed_index": round(min(1.0, point["velocity"] / 1200), 2),
                "confidence": point["confidence"],
            }
            for point in tracks[::interval][:12]
        ]
    return events[:80]


def changed_direction(points: list[dict]) -> bool:
    first_dx = points[-2]["x"] - points[-4]["x"]
    second_dx = points[-1]["x"] - points[-2]["x"]
    first_dy = points[-2]["y"] - points[-4]["y"]
    second_dy = points[-1]["y"] - points[-2]["y"]
    return (first_dx * second_dx < -120) or (first_dy * second_dy < -90)


def classify_shot(point: dict) -> str:
    y = point["y"]
    speed = point["velocity"]
    if speed > 900:
        return "drive"
    if y < 0.38:
        return "deep return"
    if y > 0.68:
        return "baseline exchange"
    return "kitchen reset"


def classify_zone(point: dict) -> str:
    x = point["x"]
    y = point["y"]
    horizontal = "left" if x < 0.33 else "right" if x > 0.67 else "middle"
    vertical = "near baseline" if y > 0.66 else "far baseline" if y < 0.34 else "kitchen"
    return f"{horizontal} {vertical}"


def build_rally_windows(events: list[dict], duration_seconds: float) -> list[dict]:
    if not events:
        return [
            {
                "label": "Full match",
                "start": 0,
                "end": round(duration_seconds, 1),
                "shots": 0,
                "note": "No reliable shot events were detected.",
            }
        ]

    windows = []
    window_size = max(60.0, duration_seconds / 6 if duration_seconds else 60.0)
    start = 0.0
    index = 1
    while start < duration_seconds:
        end = min(duration_seconds, start + window_size)
        window_events = [event for event in events if start <= event["time"] < end]
        dominant = dominant_type(window_events)
        windows.append(
            {
                "label": f"Match block {index}",
                "start": round(start, 1),
                "end": round(end, 1),
                "shots": len(window_events),
                "dominant_shot": dominant,
                "note": block_note(dominant, len(window_events)),
            }
        )
        start = end
        index += 1
    return windows


def summarize_shot_mix(events: list[dict], width: int, height: int) -> dict:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    total = sum(counts.values()) or 1
    return {
        "counts": counts,
        "shares": {label: round(count / total, 2) for label, count in counts.items()},
        "court_size": {"width": width, "height": height},
    }


def dominant_type(events: list[dict]) -> str:
    counts: dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    if not counts:
        return "none"
    return max(counts, key=counts.get)


def block_note(dominant: str, count: int) -> str:
    if count == 0:
        return "Low-confidence block; review the video manually."
    if dominant == "drive":
        return "Higher-pace exchanges detected in this block."
    if dominant == "kitchen reset":
        return "Soft-game patterns show up most in this block."
    if dominant == "deep return":
        return "Deep recovery and return patterns are prominent here."
    return "Baseline exchanges are the main detected pattern."
