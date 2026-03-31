from __future__ import annotations

"""
End-to-end GNN experiment script.

Trains the temporal GNN on benign windows, then runs inference on test windows
and prints detected alert payloads as JSON.

Usage
-----
    python scripts/run_gnn_experiment.py data/processed/cadets_e3.csv \\
        --epochs 30 \\
        --window-minutes 15 \\
        --benign-ratio 0.7 \\
        --threshold-k 3.0 \\
        --checkpoint artifacts/models/gnn_detector.pt

Requires:  pip install -e ".[ml]"
"""

import argparse
import json
from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.detection.subgraph import build_reduced_subgraph
from soc_graph.model.pipeline import run_gnn_experiment
from soc_graph.model.runtime import check_torch_backend
from soc_graph.report.serialize import serialize_reduced_subgraph


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate the GNN detector.")
    p.add_argument("input_csv", type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window-minutes", type=int, default=15)
    p.add_argument("--benign-ratio", type=float, default=0.7)
    p.add_argument("--threshold-k", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/gnn_detector.pt"),
    )
    return p.parse_args()


def main() -> None:
    backend = check_torch_backend()
    if not backend.available:
        raise RuntimeError(
            "torch / torch-geometric not available in this environment.\n"
            f"Detail: {backend.detail}\n"
            "Install with:  pip install -e \".[ml]\"\n"
            "Or use run_baseline_experiment.py instead."
        )

    args = parse_args()

    events = load_events(args.input_csv)
    snapshot_ds, artifact_ds = build_datasets(
        events=events,
        window=timedelta(minutes=args.window_minutes),
    )

    result = run_gnn_experiment(
        snapshot_dataset=snapshot_ds,
        artifact_dataset=artifact_ds,
        benign_ratio=args.benign_ratio,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold_k=args.threshold_k,
        checkpoint_path=str(args.checkpoint),
    )

    alert_payloads = []
    for idx, (snapshot, flagged_scores) in enumerate(
        zip(result.test_snapshots, result.flagged_windows, strict=True)
    ):
        if not flagged_scores:
            continue
        alert_id = f"gnn-alert-{idx + 1:04d}"
        reduced = build_reduced_subgraph(
            snapshot=snapshot,
            flagged_scores=flagged_scores,
            alert_id=alert_id,
        )
        alert_payloads.append(serialize_reduced_subgraph(reduced))

    output = {
        "input_csv": str(args.input_csv),
        "window_minutes": args.window_minutes,
        "num_events": len(events),
        "num_windows": len(snapshot_ds),
        "gnn_training": {
            "epochs": result.epochs,
            "final_loss": result.final_loss,
            "learned_threshold": result.learned_threshold,
            "checkpoint_path": result.checkpoint_path,
        },
        "graph_stats": result.graph_stats,
        "flagged_window_count": sum(1 for fw in result.flagged_windows if fw),
        "alerts": alert_payloads,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
