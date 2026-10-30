"""Verified model loading and framework-independent prediction service."""

import hashlib
import json
from pathlib import Path

import joblib

from persistency.schema import FEATURES, LABELS, validate_records


class PredictionService:
    """Load one trusted deployment artifact and reuse it across requests."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.report = json.loads((model_dir / "metrics.json").read_text())
        path = model_dir / "pipeline.joblib"
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != self.report["artifact_sha256"]
        ):
            raise ValueError("Model artifact integrity check failed.")
        self.model = joblib.load(path)  # Only trusted, reviewed repository artifacts.
        if list(self.model.feature_names_in_) != list(FEATURES):
            raise ValueError("Model feature contract does not match the application.")
        self.sample = json.loads((model_dir / "sample.json").read_text())

    def predict(self, records):
        """Return source categories and uncalibrated model class scores."""
        frame = validate_records(records)
        scores = self.model.predict_proba(frame)
        labels = self.model.predict(frame)
        return [
            {
                "category": str(label),
                "description": LABELS[str(label)],
                "scores": {
                    str(key): round(float(value), 6)
                    for key, value in zip(self.model.classes_, row)
                },
            }
            for label, row in zip(labels, scores)
        ]
