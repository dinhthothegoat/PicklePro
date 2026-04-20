"""
Deep analysis module — trains real ML models at import time on synthetic
pickleball data and exposes run_deep_analysis() for use by main.py.

Models:
  - RandomForestClassifier  (scikit-learn)
  - SVC with RBF kernel     (scikit-learn)
  - KMeans clustering       (scikit-learn, unsupervised — used for segmentation)
  - NeuralNet               (pure numpy, 8→16→8→3 feedforward)
"""

from __future__ import annotations

import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

SEED = 42
_LABELS = ["Beginner", "Intermediate", "Advanced"]


# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _generate_synthetic_data():
    """Generate 900 labeled samples (300 per class) with 8 features."""
    rng = np.random.default_rng(SEED)

    # Per-class Gaussian means for each of the 8 features
    means = np.array([
        # file_sz  dur    tempo  consist  pressure  issue_inv  opp    match_type
        [0.30,    0.25,  0.35,  0.55,    0.30,     0.70,      0.10,  0.40],  # Beginner
        [0.50,    0.50,  0.55,  0.70,    0.48,     0.40,      0.50,  0.50],  # Intermediate
        [0.70,    0.75,  0.72,  0.82,    0.65,     0.20,      0.90,  0.55],  # Advanced
    ])
    stds = np.array([0.12, 0.10, 0.08, 0.07, 0.09, 0.12, 0.05, 0.30])

    X_parts, y_parts = [], []
    for cls_idx, mu in enumerate(means):
        samples = rng.normal(loc=mu, scale=stds, size=(300, 8))
        samples = np.clip(samples, 0.0, 1.0)
        X_parts.append(samples)
        y_parts.append(np.full(300, cls_idx, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


# ---------------------------------------------------------------------------
# Feature vector builder
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _build_feature_vector(features: dict, details: dict) -> np.ndarray:
    """Map existing extract_match_features() output + details → ndarray[8]."""
    opponent_map = {"Beginner": 0.1, "Intermediate": 0.5, "Advanced": 0.9}
    vec = [
        _clamp(features.get("file_size_mb", 0.0) / 100.0),
        _clamp(features.get("estimated_duration_min", 0.0) / 45.0),
        _clamp(features.get("tempo_score", 0.5)),
        _clamp(features.get("consistency_score", 0.5)),
        _clamp(features.get("pressure_score", 0.5)),
        _clamp(1.0 - features.get("issue_complexity", 0) / 6.0),
        opponent_map.get(details.get("opponent_level", "Intermediate"), 0.5),
        0.0 if details.get("match_type", "Singles") == "Singles" else 1.0,
    ]
    return np.array(vec, dtype=float)


# ---------------------------------------------------------------------------
# Pure-numpy neural network (8 → 16 → 8 → 3)
# ---------------------------------------------------------------------------

class NeuralNet:
    def __init__(self, seed: int = SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((8, 16)) * math.sqrt(2.0 / 8)
        self.b1 = np.zeros(16)
        self.W2 = rng.standard_normal((16, 8)) * math.sqrt(2.0 / 16)
        self.b2 = np.zeros(8)
        self.W3 = rng.standard_normal((8, 3)) * math.sqrt(1.0 / 8)
        self.b3 = np.zeros(3)

    def _relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)
        z3 = a2 @ self.W3 + self.b3
        probs = self._softmax(z3)
        return probs, (X, z1, a1, z2, a2)

    def _backward(self, probs: np.ndarray, y: np.ndarray, cache, lr: float):
        X, z1, a1, z2, a2 = cache
        n = len(y)

        dz3 = probs.copy()
        dz3[np.arange(n), y] -= 1
        dz3 /= n
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W3 -= lr * dW3; self.b3 -= lr * db3
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        self.W1 -= lr * dW1; self.b1 -= lr * db1

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 200, lr: float = 0.05, batch_size: int = 64):
        rng = np.random.default_rng(SEED)
        n = len(X)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                batch = idx[start:start + batch_size]
                probs, cache = self.forward(X[batch])
                self._backward(probs, y[batch], cache, lr)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(x.reshape(1, -1))
        return probs[0]

    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        probs = self.predict_proba(x)
        idx = int(np.argmax(probs))
        return {
            "label": _LABELS[idx],
            "confidence": round(float(probs[idx]), 3),
            "probabilities": {lbl: round(float(p), 3) for lbl, p in zip(_LABELS, probs)},
        }


# ---------------------------------------------------------------------------
# Sklearn model trainers
# ---------------------------------------------------------------------------

def _train_random_forest(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    clf.fit(Xs, y)
    return clf, scaler


def _train_svm(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = SVC(kernel="rbf", probability=True, C=1.0, random_state=SEED)
    clf.fit(Xs, y)
    return clf, scaler


def _train_kmeans(X: np.ndarray, y: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED)
    km.fit(Xs)
    # Map cluster index → majority skill label
    label_map: Dict[int, str] = {}
    for c in range(3):
        mask = km.labels_ == c
        if mask.any():
            majority = int(np.bincount(y[mask]).argmax())
            label_map[c] = _LABELS[majority]
        else:
            label_map[c] = _LABELS[0]
    return km, scaler, label_map


def _sklearn_predict(clf, scaler: StandardScaler, x: np.ndarray) -> Dict[str, Any]:
    Xs = scaler.transform(x.reshape(1, -1))
    probs = clf.predict_proba(Xs)[0]
    idx = int(np.argmax(probs))
    return {
        "label": _LABELS[idx],
        "confidence": round(float(probs[idx]), 3),
        "probabilities": {lbl: round(float(p), 3) for lbl, p in zip(_LABELS, probs)},
    }


# ---------------------------------------------------------------------------
# Lazy model initialization (deferred until first call to run_deep_analysis)
# ---------------------------------------------------------------------------

_models_initialized = False
_rf_model = _rf_scaler = None
_svm_model = _svm_scaler = None
_km_model = _km_scaler = _km_label_map = None
_nn_model = None


def _ensure_models() -> None:
    global _models_initialized, _rf_model, _rf_scaler, _svm_model, _svm_scaler
    global _km_model, _km_scaler, _km_label_map, _nn_model
    if _models_initialized:
        return
    X_train, y_train = _generate_synthetic_data()
    _rf_model, _rf_scaler = _train_random_forest(X_train, y_train)
    _svm_model, _svm_scaler = _train_svm(X_train, y_train)
    _km_model, _km_scaler, _km_label_map = _train_kmeans(X_train, y_train)
    _nn_model = NeuralNet(seed=SEED)
    _nn_model.train(X_train, y_train, epochs=200, lr=0.05, batch_size=64)
    _models_initialized = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_deep_analysis(features: dict, details: dict) -> Dict[str, Any]:
    """
    Run all four real ML models and return structured ensemble result.

    Ensemble weights: RF=0.40, SVM=0.35, NN=0.25
    KMeans is unsupervised — used for segmentation only, not skill voting.
    """
    _ensure_models()
    x = _build_feature_vector(features, details)

    rf_result  = _sklearn_predict(_rf_model, _rf_scaler, x)
    svm_result = _sklearn_predict(_svm_model, _svm_scaler, x)
    nn_result  = _nn_model.predict(x)

    # KMeans segment
    Xs = _km_scaler.transform(x.reshape(1, -1))
    cluster = int(_km_model.predict(Xs)[0])
    km_result = {"segment": cluster, "label": _km_label_map[cluster]}

    # Weighted probability ensemble
    weights = [0.40, 0.35, 0.25]
    ensemble_probs = np.zeros(3)
    for w, result in zip(weights, [rf_result, svm_result, nn_result]):
        for i, lbl in enumerate(_LABELS):
            ensemble_probs[i] += w * result["probabilities"][lbl]

    ens_idx = int(np.argmax(ensemble_probs))
    ens_label = _LABELS[ens_idx]

    # Agreement: fraction of 3 voting models matching ensemble winner
    votes = [rf_result["label"], svm_result["label"], nn_result["label"]]
    agreement = round(sum(v == ens_label for v in votes) / 3.0, 3)

    return {
        "models": {
            "random_forest":  rf_result,
            "svm":            svm_result,
            "neural_net":     nn_result,
            "kmeans_segment": km_result,
        },
        "ensemble": {
            "label":         ens_label,
            "confidence":    round(float(ensemble_probs[ens_idx]), 3),
            "probabilities": {lbl: round(float(p), 3) for lbl, p in zip(_LABELS, ensemble_probs)},
            "agreement_pct": agreement,
        },
        "feature_vector": [round(float(v), 4) for v in x],
    }
