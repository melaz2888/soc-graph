from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.detection.subgraph import extract_candidate_subgraph
from soc_graph.model.io import load_detector
from soc_graph.model.train import detect_anomalies
from soc_graph.report.serialize import serialize_alert_subgraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a saved behavioral detector on a PIDSMaker-style CSV export."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("model_json", type=Path)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--window-minutes", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.input_csv)
    snapshot_dataset, artifact_dataset = build_datasets(events, window=timedelta(minutes=args.window_minutes))
    detector = load_detector(args.model_json)
    flagged_windows = detect_anomalies(detector, artifact_dataset.artifacts, threshold=args.threshold)

    alerts = []
    for snapshot_index, (snapshot, flagged_scores) in enumerate(
        zip(snapshot_dataset.snapshots, flagged_windows, strict=True)
    ):
        if not flagged_scores:
            continue
        candidate = extract_candidate_subgraph(
            snapshot=snapshot,
            flagged_scores=flagged_scores,
            alert_id=f"saved-model-alert-{snapshot_index + 1:03d}",
        )
        alerts.append(serialize_alert_subgraph(snapshot, candidate))

    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "model_json": str(args.model_json),
                "threshold": args.threshold,
                "num_windows": len(snapshot_dataset),
                "flagged_windows": [len(window_scores) for window_scores in flagged_windows],
                "alerts": alerts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

