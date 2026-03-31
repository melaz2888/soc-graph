"""Generate an executed-results notebook for the real CADETS E3 sample slice."""

from __future__ import annotations

import json
from pathlib import Path

cells: list[dict] = []


def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src,
        }
    )


md(
    """# CADETS E3 Results So Far

This notebook uses a real sample exported from the PIDSMaker `cadets_e3.dump` PostgreSQL dataset.

What this notebook demonstrates:

1. Real PIDSMaker -> PostgreSQL -> CSV ingestion
2. Time-windowed provenance graph construction
3. Baseline behavioral anomaly detection
4. GNN training on the CADETS sample slice
5. GNN loss curve and learned threshold
6. Flagged windows and top anomalous edges
7. Current LLM status
"""
)

code(
    """from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "src"))

import json
from datetime import timedelta

import matplotlib.pyplot as plt

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.detection.subgraph import build_reduced_subgraph
from soc_graph.model.gnn_inference import load_gnn_detector, score_windows
from soc_graph.model.pipeline import run_baseline_experiment, run_gnn_experiment
from soc_graph.report.llm_report import config_from_env, generate_report
from soc_graph.report.mitre_mapping import map_subgraph
from soc_graph.report.serialize import serialize_reduced_subgraph

CSV_PATH = ROOT / "data" / "processed" / "cadets_e3_sample.csv"
CHECKPOINT_PATH = ROOT / "artifacts" / "models" / "gnn_detector.pt"

assert CSV_PATH.exists(), f"Missing sample export: {CSV_PATH}"
"""
)

code(
    """events = load_events(CSV_PATH)
snapshot_ds, artifact_ds = build_datasets(events, window=timedelta(minutes=15))

window_rows = []
for idx, snapshot in enumerate(snapshot_ds.snapshots):
    window_rows.append(
        {
            "window_index": idx,
            "window_start": snapshot.window_start.isoformat(),
            "window_end": snapshot.window_end.isoformat(),
            "nodes": len(snapshot.nodes),
            "agg_edges": len(snapshot.edges),
            "raw_observations": snapshot.total_edge_observations,
        }
    )

print(f"CSV path: {CSV_PATH}")
print(f"Events loaded: {len(events):,}")
print(f"Time windows: {len(snapshot_ds):,}")
print()
for row in window_rows:
    print(
        f"Window {row['window_index']}: "
        f"{row['window_start']} -> {row['window_end']} | "
        f"nodes={row['nodes']}, agg_edges={row['agg_edges']}, raw_obs={row['raw_observations']}"
    )
"""
)

code(
    """x = [row["window_index"] for row in window_rows]
nodes = [row["nodes"] for row in window_rows]
agg_edges = [row["agg_edges"] for row in window_rows]
raw_obs = [row["raw_observations"] for row in window_rows]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].bar(x, nodes, color="#5DADE2")
axes[0].set_title("Nodes per window")
axes[0].set_xlabel("Window")

axes[1].bar(x, agg_edges, color="#52BE80")
axes[1].set_title("Aggregated edges per window")
axes[1].set_xlabel("Window")

axes[2].bar(x, raw_obs, color="#F5B041")
axes[2].set_title("Raw observations per window")
axes[2].set_xlabel("Window")

fig.suptitle("CADETS E3 sample graph structure")
plt.tight_layout()
plt.show()
"""
)

md("## Baseline detector")

code(
    """baseline_result = run_baseline_experiment(
    snapshot_ds,
    artifact_ds,
    benign_ratio=0.7,
    threshold_k=3.0,
)

baseline_counts = [len(flagged) for flagged in baseline_result.flagged_windows]

print("Baseline training summary")
print(json.dumps(
    {
        "num_windows": baseline_result.training_summary.num_windows,
        "mean_edges_per_window": baseline_result.training_summary.mean_edges_per_window,
        "learned_threshold": baseline_result.training_summary.learned_threshold,
        "benign_score_count": baseline_result.training_summary.benign_score_count,
    },
    indent=2,
))
print()
print("Flagged edge counts per test window:", baseline_counts)
"""
)

code(
    """plt.figure(figsize=(6, 4))
plt.bar(range(len(baseline_counts)), baseline_counts, color="#EC7063")
plt.title("Baseline flagged edges per test window")
plt.xlabel("Test window index")
plt.ylabel("Flagged aggregated edges")
plt.tight_layout()
plt.show()
"""
)

md("## GNN detector")

code(
    """gnn_result = run_gnn_experiment(
    snapshot_ds,
    artifact_ds,
    benign_ratio=0.7,
    epochs=5,
    threshold_k=3.0,
    checkpoint_path=str(CHECKPOINT_PATH),
)

gnn_counts = [len(flagged) for flagged in gnn_result.flagged_windows]

print("GNN training summary")
print(json.dumps(
    {
        "epochs": gnn_result.epochs,
        "final_loss": gnn_result.final_loss,
        "loss_history": gnn_result.loss_history,
        "learned_threshold": gnn_result.learned_threshold,
        "checkpoint_path": gnn_result.checkpoint_path,
    },
    indent=2,
))
print()
print("Flagged edge counts per test window:", gnn_counts)
"""
)

code(
    """plt.figure(figsize=(7, 4))
plt.plot(range(1, len(gnn_result.loss_history) + 1), gnn_result.loss_history, marker="o", color="#8E44AD")
plt.title("GNN training loss by epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean BCE loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

code(
    """fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].bar(range(len(baseline_counts)), baseline_counts, color="#EC7063")
axes[0].set_title("Baseline flagged edges")
axes[0].set_xlabel("Test window")
axes[0].set_ylabel("Flagged edges")

axes[1].bar(range(len(gnn_counts)), gnn_counts, color="#AF7AC5")
axes[1].set_title("GNN flagged edges")
axes[1].set_xlabel("Test window")
axes[1].set_ylabel("Flagged edges")

plt.tight_layout()
plt.show()
"""
)

md("## Top anomalous edges from the GNN")

code(
    """detector = load_gnn_detector(CHECKPOINT_PATH)
_, test_artifacts = artifact_ds.train_test_split(benign_ratio=0.7)
gnn_scores = score_windows(detector, test_artifacts)

top_edges = []
for window_idx, window_scores in enumerate(gnn_scores):
    for edge_key, score in window_scores.items():
        top_edges.append((score, window_idx, edge_key))

top_edges = sorted(top_edges, reverse=True)[:20]
for rank, (score, window_idx, edge_key) in enumerate(top_edges, start=1):
    print(f"{rank:02d}. window={window_idx}  score={score:.4f}  edge={edge_key}")
"""
)

code(
    """alerts = []
for idx, (snapshot, flagged) in enumerate(zip(gnn_result.test_snapshots, gnn_result.flagged_windows, strict=True)):
    if not flagged:
        continue
    alert_id = f"cadets-sample-gnn-alert-{idx + 1:03d}"
    reduced = build_reduced_subgraph(snapshot=snapshot, flagged_scores=flagged, alert_id=alert_id)
    alerts.append(serialize_reduced_subgraph(reduced))

print(f"GNN produced {len(alerts)} serialized alert(s).")
if alerts:
    first_alert = alerts[0]
    print(json.dumps(
        {
            "alert_id": first_alert["alert_id"],
            "flagged_edge_count": first_alert["flagged_edge_count"],
            "total_edge_count": first_alert["total_edge_count"],
            "component_count": first_alert["component_count"],
            "num_nodes": len(first_alert["nodes"]),
            "num_edges": len(first_alert["edges"]),
        },
        indent=2,
    ))
"""
)

md("## MITRE and report layer")

code(
    """if alerts:
    edge_hints = [
        {
            "src_type": edge["src_type"],
            "edge_type": edge["type"],
            "dst_type": edge["dst_type"],
        }
        for edge in alerts[0]["edges"]
    ]
    mitre = map_subgraph(edge_hints)
    print("Mapped ATT&CK techniques:")
    for item in mitre[:10]:
        print(f"- {item['technique_id']} {item['technique_name']} [{item['tactic']}]")
else:
    print("No alerts available for ATT&CK mapping.")
"""
)

code(
    """llm_cfg = config_from_env()
print("LLM configured:" , bool(llm_cfg))
if llm_cfg:
    llm_cfg.timeout = max(llm_cfg.timeout, 600)
    print(f"Provider: {llm_cfg.provider}")
    print(f"Model: {llm_cfg.model}")
else:
    print("No live LLM is configured in this environment.")
    print("The report layer currently falls back to a deterministic placeholder.")

if alerts:
    report = generate_report(alerts[0], config=llm_cfg)
    print()
    print("Report verdict:", report.verdict)
    print("Confidence:", report.confidence)
    print("Narrative:", report.narrative)
    print("Note:", report.note)
"""
)

md(
    """## Interpretation

What we can concretely do so far:

- ingest real PIDSMaker CADETS E3 data through PostgreSQL export
- build time-windowed provenance graphs
- run a baseline structural anomaly detector
- run the first GNN path on a real CADETS-derived sample slice
- serialize suspicious subgraphs and map them to MITRE ATT&CK hints
- generate an analyst report through Ollama when configured, otherwise fall back to a deterministic placeholder

What is not set up yet:

- no full-dataset CADETS E3 run over all 36M+ events yet
- no polished evaluation benchmark yet
"""
)


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

out = Path(__file__).parent.parent / "notebooks" / "03_cadets_e3_results.ipynb"
out.parent.mkdir(exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    json.dump(notebook, handle, indent=1)

print(f"Written: {out}")
