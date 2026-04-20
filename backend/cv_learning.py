"""Trainable computer-vision scoring layer for pickleball match videos.

The live app can run without a trained artifact. When a model artifact exists,
the frame analyzer blends its learned scores with the rule-based CV signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "avg_motion",
    "motion_variance",
    "motion_burst_rate",
    "avg_brightness",
    "avg_sharpness",
    "court_presence",
    "net_activity_ratio",
    "baseline_activity_ratio",
    "lateral_balance",
    "duration_norm",
    "resolution_norm",
]

TARGET_NAMES = ["tempo_score", "consistency_score", "pressure_score"]


@dataclass
class VisionPrediction:
    tempo_score: float
    consistency_score: float
    pressure_score: float
    confidence: float
    training_samples: int
    artifact_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tempo_score": round(self.tempo_score, 3),
            "consistency_score": round(self.consistency_score, 3),
            "pressure_score": round(self.pressure_score, 3),
            "confidence": round(self.confidence, 3),
            "training_samples": self.training_samples,
            "artifact_version": self.artifact_version,
        }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def metrics_to_vector(metrics: dict[str, Any]) -> np.ndarray:
    width = float(metrics.get("width") or 0)
    height = float(metrics.get("height") or 0)
    duration = float(metrics.get("duration_seconds") or 0)
    vector = [
        float(metrics.get("avg_motion") or 0),
        float(metrics.get("motion_variance") or 0),
        float(metrics.get("motion_burst_rate") or 0),
        float(metrics.get("avg_brightness") or 0),
        float(metrics.get("avg_sharpness") or 0),
        float(metrics.get("court_presence") or 0),
        float(metrics.get("net_activity_ratio") or 0),
        float(metrics.get("baseline_activity_ratio") or 0),
        float(metrics.get("lateral_balance") or 0),
        clamp(duration / 3600.0),
        clamp((width * height) / (1920.0 * 1080.0)),
    ]
    return np.array(vector, dtype=float)


def heuristic_targets(metrics: dict[str, Any]) -> np.ndarray:
    """Build self-supervised labels from CV signals when human labels are absent."""
    motion = float(metrics.get("avg_motion") or 0)
    variance = float(metrics.get("motion_variance") or 0)
    burst = float(metrics.get("motion_burst_rate") or 0)
    sharpness = float(metrics.get("avg_sharpness") or 0)
    court = float(metrics.get("court_presence") or 0)
    net = float(metrics.get("net_activity_ratio") or 0)
    lateral = float(metrics.get("lateral_balance") or 0.5)

    tempo = clamp(motion * 4.8 + variance * 18 + burst * 0.22)
    consistency = clamp(0.88 - variance * 22 - burst * 0.12 + sharpness * 0.05 + lateral * 0.05)
    pressure = clamp(0.34 + tempo * 0.38 + court * 0.18 + net * 0.08)
    return np.array([tempo, consistency, pressure], dtype=float)


def _confidence_from_training_size(training_samples: int) -> float:
    # A smooth ramp: useful after dozens, strong around hundreds, high near 3000.
    return clamp(0.25 + np.log1p(training_samples) / np.log1p(3000) * 0.65)


def train_cv_artifact(samples: list[dict[str, Any]], artifact_path: str | Path) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one analyzed video sample is required.")

    X = np.vstack([metrics_to_vector(sample["metrics"]) for sample in samples])
    y = np.vstack([np.array(sample.get("targets") or heuristic_targets(sample["metrics"])) for sample in samples])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestRegressor(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(Xs, y)

    artifact = {
        "version": "picklecoach-cv-rf-v1",
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "training_samples": len(samples),
        "scaler": scaler,
        "model": model,
    }
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as f:
        pickle.dump(artifact, f)
    return {
        "artifact_path": str(artifact_path),
        "training_samples": len(samples),
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }


def load_cv_artifact(artifact_path: str | Path) -> dict[str, Any] | None:
    path = Path(artifact_path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        artifact = pickle.load(f)
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("CV model artifact feature schema does not match this app version.")
    return artifact


def predict_with_artifact(metrics: dict[str, Any], artifact: dict[str, Any]) -> VisionPrediction:
    vector = metrics_to_vector(metrics).reshape(1, -1)
    scaled = artifact["scaler"].transform(vector)
    tempo, consistency, pressure = artifact["model"].predict(scaled)[0]
    training_samples = int(artifact.get("training_samples") or 0)
    return VisionPrediction(
        tempo_score=clamp(float(tempo)),
        consistency_score=clamp(float(consistency)),
        pressure_score=clamp(float(pressure)),
        confidence=_confidence_from_training_size(training_samples),
        training_samples=training_samples,
        artifact_version=str(artifact.get("version") or "unknown"),
    )
