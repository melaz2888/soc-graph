from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.model.gnn_train import GNNTrainingConfig, train_gnn_detector
from soc_graph.model.runtime import check_torch_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first GNN detector on benign provenance windows.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--window-minutes", type=int, default=15)
    parser.add_argument("--benign-ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/models/gnn_detector.pt"),
    )
    return parser.parse_args()


def main() -> None:
    backend = check_torch_backend()
    if not backend.available:
        raise RuntimeError(
            "Torch backend unavailable in this environment. "
            f"GNN code path is implemented but cannot run here yet: {backend.detail}"
        )

    args = parse_args()
    events = load_events(args.input_csv)
    _, artifact_dataset = build_datasets(events, window=timedelta(minutes=args.window_minutes))
    train_artifacts, _ = artifact_dataset.train_test_split(benign_ratio=args.benign_ratio)

    summary = train_gnn_detector(
        train_artifacts,
        training_config=GNNTrainingConfig(
            epochs=args.epochs,
            checkpoint_path=str(args.checkpoint_path),
        ),
    )
    print(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "epochs": summary.epochs,
                "final_loss": summary.final_loss,
                "checkpoint_path": summary.checkpoint_path,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

