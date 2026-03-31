from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.detection.subgraph import extract_candidate_subgraph
from soc_graph.model.pipeline import run_baseline_experiment
from soc_graph.report.serialize import serialize_alert_subgraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline SOC-Graph experiment on a PIDSMaker-style CSV export."
    )
    parser.add_argument("input_csv", type=Path, help="Path to a PIDSMaker-style CSV export.")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="Time window size in minutes.",
    )
    parser.add_argument(
        "--benign-ratio",
        type=float,
        default=0.7,
        help="Fraction of earliest windows treated as benign training data.",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=3.0,
        help="Sigma multiplier used for anomaly threshold calibration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.input_csv)
    snapshot_dataset, artifact_dataset = build_datasets(
        events=events,
        window=timedelta(minutes=args.window_minutes),
    )

    result = run_baseline_experiment(
        snapshot_dataset=snapshot_dataset,
        artifact_dataset=artifact_dataset,
        benign_ratio=args.benign_ratio,
        threshold_k=args.threshold_k,
    )

    alert_payloads = []
    for snapshot_index, (snapshot, flagged_scores) in enumerate(
        zip(result.test_snapshots, result.flagged_windows, strict=True)
    ):
        if not flagged_scores:
            continue
        candidate = extract_candidate_subgraph(
            snapshot=snapshot,
            flagged_scores=flagged_scores,
            alert_id=f"baseline-alert-{snapshot_index + 1:03d}",
        )
        alert_payloads.append(serialize_alert_subgraph(snapshot, candidate))

    output = {
        "input_csv": str(args.input_csv),
        "window_minutes": args.window_minutes,
        "num_events": len(events),
        "num_windows": len(snapshot_dataset),
        "training_summary": {
            "num_windows": result.training_summary.num_windows,
            "mean_edges_per_window": result.training_summary.mean_edges_per_window,
            "learned_threshold": result.training_summary.learned_threshold,
            "benign_score_count": result.training_summary.benign_score_count,
        },
        "graph_stats": result.graph_stats,
        "flagged_windows": [len(window_scores) for window_scores in result.flagged_windows],
        "alerts": alert_payloads,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

