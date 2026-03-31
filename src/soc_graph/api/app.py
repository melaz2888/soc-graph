from __future__ import annotations

"""
FastAPI application for the SOC-Graph detection pipeline.

Startup:
    The app reads SOC_GRAPH_DATA_CSV (required) and optional tuning knobs
    from environment variables, runs the detection pipeline once, then serves
    the results through REST endpoints.

    uvicorn soc_graph.api.app:app --reload

Environment variables:
    SOC_GRAPH_DATA_CSV      Path to a PIDSMaker-style CSV export (required).
    SOC_GRAPH_WINDOW_MIN    Time window in minutes (default: 15).
    SOC_GRAPH_BENIGN_RATIO  Fraction of early windows used as benign training (default: 0.7).
    SOC_GRAPH_THRESHOLD_K   Sigma multiplier for anomaly threshold (default: 3.0).
    OLLAMA_BASE_URL         Ollama server URL (optional, enables LLM reports).
    OLLAMA_MODEL            Ollama model name (default: qwen2.5).
    OPENAI_API_KEY          OpenAI key (optional, takes priority over Ollama).
    OPENAI_MODEL            OpenAI model (default: gpt-4o-mini).
"""

import os
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

from soc_graph.data.dataset import build_datasets
from soc_graph.data.pidsmaker import load_events
from soc_graph.detection.subgraph import build_reduced_subgraph
from soc_graph.model.pipeline import run_baseline_experiment
from soc_graph.report.llm_report import LLMConfig, config_from_env, generate_report
from soc_graph.report.serialize import serialize_reduced_subgraph


# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

class _AppState:
    alerts: dict[str, dict[str, Any]] = {}
    reports: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    config: dict[str, Any] = {}
    llm_config: LLMConfig | None = None


_state = _AppState()


def _run_pipeline() -> None:
    csv_path = os.environ.get("SOC_GRAPH_DATA_CSV", "")
    if not csv_path:
        _state.metrics = {"error": "SOC_GRAPH_DATA_CSV not set; no data loaded."}
        return

    window_min = int(os.environ.get("SOC_GRAPH_WINDOW_MIN", "15"))
    benign_ratio = float(os.environ.get("SOC_GRAPH_BENIGN_RATIO", "0.7"))
    threshold_k = float(os.environ.get("SOC_GRAPH_THRESHOLD_K", "3.0"))

    _state.config = {
        "csv_path": csv_path,
        "window_minutes": window_min,
        "benign_ratio": benign_ratio,
        "threshold_k": threshold_k,
    }

    events = load_events(csv_path)
    snapshot_dataset, artifact_dataset = build_datasets(
        events=events, window=timedelta(minutes=window_min)
    )

    result = run_baseline_experiment(
        snapshot_dataset=snapshot_dataset,
        artifact_dataset=artifact_dataset,
        benign_ratio=benign_ratio,
        threshold_k=threshold_k,
    )

    _state.metrics = {
        "num_events": len(events),
        "num_windows": len(snapshot_dataset),
        "training": {
            "num_windows": result.training_summary.num_windows,
            "mean_edges_per_window": result.training_summary.mean_edges_per_window,
            "learned_threshold": result.training_summary.learned_threshold,
            "benign_score_count": result.training_summary.benign_score_count,
        },
        "graph_stats": result.graph_stats,
        "flagged_window_count": sum(1 for fw in result.flagged_windows if fw),
    }

    _state.llm_config = config_from_env()

    for idx, (snapshot, flagged_scores) in enumerate(
        zip(result.test_snapshots, result.flagged_windows, strict=True)
    ):
        if not flagged_scores:
            continue
        alert_id = f"alert-{idx + 1:04d}"
        reduced = build_reduced_subgraph(
            snapshot=snapshot,
            flagged_scores=flagged_scores,
            alert_id=alert_id,
        )
        payload = serialize_reduced_subgraph(reduced)
        _state.alerts[alert_id] = payload


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):  # type: ignore[type-arg]
    _run_pipeline()
    yield


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    app = FastAPI(
        title="SOC-Graph",
        description="GNN-based APT detection on system provenance graphs.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # GET /
    # ------------------------------------------------------------------
    @app.get("/", tags=["health"])
    def root():
        """Health check and pipeline status."""
        return {
            "status": "ok",
            "alerts_loaded": len(_state.alerts),
            "llm_available": _state.llm_config is not None,
            "config": _state.config,
        }

    # ------------------------------------------------------------------
    # GET /metrics
    # ------------------------------------------------------------------
    @app.get("/metrics", tags=["metrics"])
    def metrics():
        """Return model performance and graph statistics."""
        return JSONResponse(_state.metrics)

    # ------------------------------------------------------------------
    # GET /alerts
    # ------------------------------------------------------------------
    @app.get("/alerts", tags=["alerts"])
    def list_alerts(limit: int = 50, offset: int = 0):
        """List detected alert summaries (paginated)."""
        all_ids = sorted(_state.alerts)
        page = all_ids[offset: offset + limit]
        return {
            "total": len(all_ids),
            "offset": offset,
            "limit": limit,
            "alerts": [
                {
                    "alert_id": aid,
                    "flagged_edge_count": _state.alerts[aid]["flagged_edge_count"],
                    "total_edge_count": _state.alerts[aid]["total_edge_count"],
                    "component_count": _state.alerts[aid]["component_count"],
                    "node_count": len(_state.alerts[aid]["nodes"]),
                }
                for aid in page
            ],
        }

    # ------------------------------------------------------------------
    # GET /alerts/{alert_id}
    # ------------------------------------------------------------------
    @app.get("/alerts/{alert_id}", tags=["alerts"])
    def get_alert(alert_id: str):
        """
        Return the full alert payload including the LLM investigation report.
        Reports are generated lazily and cached in memory.
        """
        if alert_id not in _state.alerts:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")

        payload = dict(_state.alerts[alert_id])

        if alert_id not in _state.reports:
            report = generate_report(payload, config=_state.llm_config)
            _state.reports[alert_id] = {
                "verdict": report.verdict,
                "confidence": report.confidence,
                "narrative": report.narrative,
                "mitre_techniques": report.mitre_techniques,
                "recommended_actions": report.recommended_actions,
                "note": report.note,
            }

        payload["report"] = _state.reports[alert_id]
        return JSONResponse(payload)

    # ------------------------------------------------------------------
    # GET /graph/{alert_id}
    # ------------------------------------------------------------------
    @app.get("/graph/{alert_id}", tags=["graph"])
    def get_graph(alert_id: str):
        """
        Return the compact attack summary graph for a specific alert.
        Suitable for rendering in a graph visualisation tool.
        """
        if alert_id not in _state.alerts:
            raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
        return JSONResponse(_state.alerts[alert_id])

    # ------------------------------------------------------------------
    # POST /analyze  (on-demand, does not mutate cached state)
    # ------------------------------------------------------------------
    @app.post("/analyze", tags=["analyze"])
    async def analyze(body: dict[str, Any]):
        """
        Run detection on a pre-built serialized alert payload and return an
        LLM investigation report.  Accepts the same JSON format returned by
        GET /graph/{alert_id}.
        """
        report = generate_report(body, config=_state.llm_config)
        return {
            "verdict": report.verdict,
            "confidence": report.confidence,
            "narrative": report.narrative,
            "mitre_techniques": report.mitre_techniques,
            "recommended_actions": report.recommended_actions,
            "note": report.note,
        }

    return app


# Module-level app instance for uvicorn
app = create_app()
