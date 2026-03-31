from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.model.detector import BehavioralAnomalyDetector
from soc_graph.model.io import save_detector
from soc_graph.model.pipeline import summarize_graph_windows
from soc_graph.model.train import fit_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the behavioral anomaly detector on a PIDSMaker-style CSV export."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the PIDSMaker-style CSV file.")
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path("artifacts/models/behavioral_detector.json"),
        help="Where to write the trained detector artifact.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("artifacts/models/behavioral_detector_summary.json"),
        help="Where to write the training summary JSON.",
    )
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--benign-ratio", type=float, default=0.7)
    parser.add_argument("--threshold-k", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.input_csv)
    snapshot_dataset, artifact_dataset = build_datasets(events, window=timedelta(minutes=args.window_minutes))
    train_snapshots, _ = snapshot_dataset.train_test_split(benign_ratio=args.benign_ratio)
    train_artifacts, _ = artifact_dataset.train_test_split(benign_ratio=args.benign_ratio)

    detector = BehavioralAnomalyDetector()
    summary = fit_detector(detector, benign_artifacts=train_artifacts, threshold_k=args.threshold_k)
    save_detector(detector, args.output_model)

    output = {
        "input_csv": str(args.input_csv),
        "output_model": str(args.output_model),
        "window_minutes": args.window_minutes,
        "benign_ratio": args.benign_ratio,
        "threshold_k": args.threshold_k,
        "training_summary": {
            "num_windows": summary.num_windows,
            "mean_edges_per_window": summary.mean_edges_per_window,
            "learned_threshold": summary.learned_threshold,
            "benign_score_count": summary.benign_score_count,
        },
        "graph_stats": summarize_graph_windows(snapshot_dataset),
        "train_windows": len(train_snapshots),
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

