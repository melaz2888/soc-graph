from __future__ import annotations

import json
from pathlib import Path

from .detector import BehavioralAnomalyDetector


def save_detector(detector: BehavioralAnomalyDetector, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(detector.to_dict(), indent=2), encoding="utf-8")


def load_detector(path: str | Path) -> BehavioralAnomalyDetector:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detector payload must be a JSON object")
    return BehavioralAnomalyDetector.from_dict(payload)

