"""Generate notebooks/02_pipeline_demo.ipynb"""
from __future__ import annotations
import json
from pathlib import Path

cells: list[dict] = []

def md(src: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src: str) -> None:
    cells.append({
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": src,
    })

# ── Cell 1: title ─────────────────────────────────────────────────────────────
md("""\
# SOC-Graph Pipeline Demo

**End-to-end APT detection on a synthetic provenance graph.**

**Scenario:** A red team exploited an nginx web server (`/usr/sbin/nginx`),
spawned `/bin/sh`, downloaded tooling from C2 at `128.55.12.167`,
and exfiltrated `/etc/passwd` and `/root/.ssh/id_rsa`.
Normal nginx / sshd / python activity runs as background noise.

This notebook runs:
1. Graph construction from raw CSV events
2. Behavioral anomaly detector *(no GPU required)*
3. GNN encoder-decoder *(temporal GAT + GRU — skipped if torch unavailable)*
4. Attack subgraph extraction
5. MITRE ATT\\&CK mapping
6. LLM investigation report *(placeholder if no Ollama/OpenAI configured)*
""")

# ── Cell 2: load data ─────────────────────────────────────────────────────────
code("""\
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "../src")

from datetime import timedelta
from soc_graph.data.pidsmaker import load_events
from soc_graph.data.dataset import build_datasets

CSV = "../data/demo/apt_scenario.csv"
events = load_events(CSV)
snap_ds, art_ds = build_datasets(events, window=timedelta(minutes=15))

print(f"Total events  : {len(events)}")
print(f"Time windows  : {len(snap_ds)}")
print()
for i, snap in enumerate(snap_ds.snapshots):
    label = "BENIGN" if i < 3 else "*** ATTACK ***"
    print(
        f"  Window {i}  {snap.window_start.strftime('%H:%M')}-{snap.window_end.strftime('%H:%M')}"
        f"  nodes={len(snap.nodes):2d}  agg-edges={len(snap.edges):2d}"
        f"  raw-obs={snap.total_edge_observations:3d}  [{label}]"
    )
""")

# ── Cell 3: edge type distribution ────────────────────────────────────────────
md("## Edge-type distribution per window")

code("""\
for i, snap in enumerate(snap_ds.snapshots):
    label = "benign" if i < 3 else "ATTACK"
    counts = snap.edge_type_counts
    parts = "  ".join(
        f"{k.value}:{v}"
        for k, v in sorted(counts.items(), key=lambda x: -x[1])
    )
    print(f"Window {i} [{label}]  {parts}")
""")

# ── Cell 4: behavioral detector ───────────────────────────────────────────────
md("## Behavioral anomaly detector (no GPU)")

code("""\
from soc_graph.model.pipeline import run_baseline_experiment

result = run_baseline_experiment(snap_ds, art_ds, benign_ratio=0.75, threshold_k=2.5)
ts = result.training_summary

print(f"Training windows  : {ts.num_windows}")
print(f"Mean edges/window : {ts.mean_edges_per_window:.1f}")
print(f"Learned threshold : {ts.learned_threshold:.4f}  (mean + 2.5 sigma of benign scores)")
print()
print("Flagged edge counts per test window:")
for i, fw in enumerate(result.flagged_windows):
    bar = "=" * min(len(fw), 50)
    print(f"  Test window {i}: {len(fw):3d} flagged  |{bar}|")
""")

# ── Cell 5: build alert payloads ──────────────────────────────────────────────
md("## Flagged attack subgraph")

code("""\
from soc_graph.detection.subgraph import build_reduced_subgraph
from soc_graph.report.serialize import serialize_reduced_subgraph

alerts = []
for i, (snap, flagged) in enumerate(zip(result.test_snapshots, result.flagged_windows)):
    if not flagged:
        continue
    aid = f"behavioral-alert-{i+1:03d}"
    reduced = build_reduced_subgraph(snap, flagged, aid)
    payload = serialize_reduced_subgraph(reduced)
    alerts.append(payload)

    print(f"Alert: {aid}")
    print(f"  Nodes in subgraph : {len(payload['nodes'])}")
    print(f"  Flagged edges     : {payload['flagged_edge_count']} / {payload['total_edge_count']} total")
    print(f"  Components        : {payload['component_count']}")
    print()
    print("  Nodes:")
    for n in payload["nodes"]:
        print(f"    [{n['type']:7s}]  {n['name']}")
    print()
    print("  Top anomalous edges:")
    top = sorted(payload["edges"], key=lambda e: e.get("anomaly_score", 0), reverse=True)[:10]
    for e in top:
        print(
            f"    {e['src_type']:7s} --{e['type']:8s}--> {e['dst_type']:7s}"
            f"  score={e.get('anomaly_score', 0):.3f}  x{e['count']}"
        )
""")

# ── Cell 6: graphviz visualisation ────────────────────────────────────────────
md("## Attack subgraph visualisation")

code("""\
if not alerts:
    print("No alerts to visualise.")
else:
    payload = alerts[0]

    def safe(s):
        for ch in "-:/.":
            s = s.replace(ch, "_")
        return s

    shapes = {"PROCESS": "ellipse", "FILE": "box", "SOCKET": "diamond"}
    colors = {"PROCESS": "#AED6F1", "FILE": "#A9DFBF", "SOCKET": "#F9E79F"}

    lines = [
        "digraph G {",
        '  rankdir="LR";',
        '  node [fontname="Helvetica" fontsize=10];',
        '  edge [fontsize=9];',
    ]
    for n in payload["nodes"]:
        nid = safe(n["id"])
        label = n["name"]
        lines.append(
            f'  {nid} [label="{label}" shape={shapes[n["type"]]} '
            f'style=filled fillcolor="{colors[n["type"]]}"];'
        )
    for e in payload["edges"]:
        src, dst = safe(e["src"]), safe(e["dst"])
        score = e.get("anomaly_score", 0)
        lines.append(
            f'  {src} -> {dst} [label="{e["type"]} x{e["count"]} score={score:.2f}"];'
        )
    lines.append("}")
    dot_src = "\\n".join(lines)

    try:
        import graphviz
        display(graphviz.Source(dot_src))
    except Exception:
        print("graphviz not installed — DOT source:")
        print(dot_src)
""")

# ── Cell 7: GNN ───────────────────────────────────────────────────────────────
md("## GNN encoder-decoder (temporal GAT + GRU)")

code("""\
from soc_graph.model.runtime import check_torch_backend

backend = check_torch_backend()
print("Torch backend:", "available" if backend.available else f"UNAVAILABLE: {backend.detail}")
""")

code("""\
if backend.available:
    from soc_graph.model.pipeline import run_gnn_experiment

    gnn_result = run_gnn_experiment(
        snap_ds, art_ds,
        benign_ratio=0.75,
        epochs=15,
        threshold_k=2.5,
        checkpoint_path="../artifacts/models/gnn_demo.pt",
    )
    print(f"Final training loss : {gnn_result.final_loss:.4f}")
    print(f"Learned threshold   : {gnn_result.learned_threshold:.4f}")
    print()
    print("Flagged edge counts per test window (GNN):")
    for i, fw in enumerate(gnn_result.flagged_windows):
        bar = "=" * min(len(fw), 50)
        print(f"  Test window {i}: {len(fw):3d} flagged  |{bar}|")

    # Build GNN alerts too
    gnn_alerts = []
    for i, (snap, flagged) in enumerate(zip(gnn_result.test_snapshots, gnn_result.flagged_windows)):
        if not flagged:
            continue
        aid = f"gnn-alert-{i+1:03d}"
        reduced = build_reduced_subgraph(snap, flagged, aid)
        gnn_alerts.append(serialize_reduced_subgraph(reduced))
    print(f"\\nGNN raised {len(gnn_alerts)} alert(s).")
else:
    gnn_alerts = []
    print("Skipping GNN — using behavioral alerts for remaining cells.")
""")

# ── Cell 8: MITRE mapping ─────────────────────────────────────────────────────
md("## MITRE ATT&CK mapping")

code("""\
from soc_graph.report.mitre_mapping import map_subgraph

source_alerts = gnn_alerts if (backend.available and gnn_alerts) else alerts

if source_alerts:
    edge_hints = [
        {
            "src_type": e["src_type"],
            "edge_type": e["type"],
            "dst_type": e["dst_type"],
        }
        for e in source_alerts[0]["edges"]
    ]
    techniques = map_subgraph(edge_hints)
    print(f"Mapped {len(techniques)} ATT&CK techniques from the flagged subgraph:\\n")
    for t in techniques:
        print(f"  {t['technique_id']}  {t['technique_name']:<45s}  [{t['tactic']}]")
        print(f"           {t['rationale']}")
        print()
""")

# ── Cell 9: report ────────────────────────────────────────────────────────────
md("## Investigation report")

code("""\
from soc_graph.report.llm_report import generate_report, config_from_env

llm_cfg = config_from_env()
if llm_cfg:
    print(f"LLM provider : {llm_cfg.provider} / {llm_cfg.model}")
else:
    print("No LLM configured — generating placeholder report.")
    print("  To enable: ollama pull qwen2.5 && ollama serve")
    print("  Or set: OPENAI_API_KEY=sk-...")
print()

source_alerts = gnn_alerts if (backend.available and gnn_alerts) else alerts

if source_alerts:
    report = generate_report(source_alerts[0], config=llm_cfg)

    icons = {"malicious": "[MALICIOUS]", "suspicious": "[SUSPICIOUS]", "benign": "[BENIGN]"}
    print(f"{icons.get(report.verdict, '[?]')} VERDICT    : {report.verdict.upper()}")
    print(f"             CONFIDENCE : {report.confidence}")
    print()
    print("NARRATIVE:")
    print(f"  {report.narrative}")
    if report.recommended_actions:
        print()
        print("RECOMMENDED ACTIONS:")
        for a in report.recommended_actions:
            print(f"  - {a}")
    if report.note:
        print()
        print(f"Note: {report.note}")
""")

# ── Cell 10: summary ──────────────────────────────────────────────────────────
md("""\
## Status summary

| Component | Status |
|-----------|--------|
| PIDSMaker CSV ingestion | working |
| Time-windowed graph construction | working |
| `GraphTensorArtifact` (PyG-ready tensors) | working |
| Behavioral anomaly detector | working (no GPU) |
| GNN encoder (2-layer GAT + GRU memory) | working (requires torch) |
| GNN edge decoder (MLP) | working |
| Threshold calibration (sigma-based) | working |
| Connected-component subgraph extraction | working |
| MITRE ATT&CK rule mapping | working |
| LLM report generation | wired — needs Ollama or OPENAI_API_KEY |
| FastAPI REST API | implemented |
| Streamlit dashboard | implemented |

**Run the API:**
```bash
SOC_GRAPH_DATA_CSV=data/demo/apt_scenario.csv uvicorn soc_graph.api.app:app --reload
```

**Run the dashboard:**
```bash
streamlit run src/soc_graph/dashboard/streamlit_app.py
```
""")

# ── Write notebook ─────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).parent.parent / "notebooks" / "02_pipeline_demo.ipynb"
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Written: {out}")
