from __future__ import annotations

"""
LLM-powered investigation report generation.

Supports two providers out of the box:
  - Ollama  (local, default): set OLLAMA_BASE_URL (default http://localhost:11434)
                               and OLLAMA_MODEL   (default qwen2.5)
  - OpenAI  (remote):         set OPENAI_API_KEY
                               and OPENAI_MODEL    (default gpt-4o-mini)

If no provider is configured the module falls back to a deterministic
placeholder so the rest of the pipeline keeps working.
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from soc_graph.report.mitre_mapping import map_subgraph


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportArtifact:
    verdict: str          # "malicious" | "suspicious" | "benign" | "placeholder"
    confidence: str       # "high" | "medium" | "low"
    narrative: str
    mitre_techniques: list[dict[str, str]] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class LLMConfig:
    provider: str = "ollama"          # "ollama" | "openai"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5"
    api_key: str | None = None
    timeout: int = 120


# ---------------------------------------------------------------------------
# Config auto-detection from environment variables
# ---------------------------------------------------------------------------

def config_from_env() -> LLMConfig | None:
    """
    Return an LLMConfig derived from environment variables, or None if
    neither Ollama nor OpenAI is configured.
    """
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return LLMConfig(
            provider="openai",
            base_url="https://api.openai.com",
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
        )

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen2.5").strip()
    # Attempt a lightweight connectivity check; return None if Ollama is not up.
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            pass
        return LLMConfig(provider="ollama", base_url=ollama_url, model=ollama_model)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior SOC analyst AI. "
    "Given an attack summary graph extracted from system provenance data by a "
    "GNN anomaly detector, produce a concise investigation report. "
    "The GNN was trained on normal system behaviour; edges with high anomaly "
    "scores represent system events the model did not expect. "
    "Be precise and factual. Do not speculate beyond what the graph shows."
)

_REPORT_FORMAT = """
Produce the report in EXACTLY this format (use the section headers verbatim):

VERDICT: <malicious | suspicious | benign>
CONFIDENCE: <high | medium | low>

ATTACK NARRATIVE:
<Step-by-step chronological description of what happened, referencing specific
process names, file paths and network addresses from the graph.>

MITRE ATT&CK MAPPING:
<List each relevant ATT&CK technique as: TID - Name (Tactic)>

RECOMMENDED ACTIONS:
<Bulleted list of concrete analyst next steps>
"""


def _build_user_message(alert_payload: dict) -> str:
    compact = json.dumps(alert_payload, indent=2, default=str)
    return f"Analyse the following alert and produce an investigation report.\n\n{compact}\n\n{_REPORT_FORMAT}"


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only – no requests/httpx dependency)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> str:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _call_ollama(prompt: str, config: LLMConfig) -> str:
    url = f"{config.base_url.rstrip('/')}/api/chat"
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    raw = _post_json(url, payload, headers, config.timeout)
    parsed = json.loads(raw)
    return parsed["message"]["content"]


def _call_openai(prompt: str, config: LLMConfig) -> str:
    url = f"{config.base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    raw = _post_json(url, payload, headers, config.timeout)
    parsed = json.loads(raw)
    return parsed["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_section(text: str, header: str, next_headers: list[str]) -> str:
    """Extract the text between `header:` and the next known section header."""
    upper = text.upper()
    start_marker = f"{header.upper()}:"
    start = upper.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = len(text)
    for nh in next_headers:
        pos = upper.find(f"{nh.upper()}:", start)
        if pos != -1 and pos < end:
            end = pos
    return text[start:end].strip()


def _parse_response(text: str) -> dict:
    headers_order = [
        "VERDICT", "CONFIDENCE", "ATTACK NARRATIVE",
        "MITRE ATT&CK MAPPING", "RECOMMENDED ACTIONS",
    ]
    result: dict[str, str] = {}
    for i, header in enumerate(headers_order):
        rest = headers_order[i + 1:]
        result[header] = _extract_section(text, header, rest)

    verdict = result.get("VERDICT", "").split("\n")[0].strip().lower()
    if verdict not in ("malicious", "suspicious", "benign"):
        verdict = "suspicious"

    confidence = result.get("CONFIDENCE", "").split("\n")[0].strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    # Parse recommended actions into a list
    actions_raw = result.get("RECOMMENDED ACTIONS", "")
    actions = [
        line.lstrip("-•* ").strip()
        for line in actions_raw.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    return {
        "verdict": verdict,
        "confidence": confidence,
        "narrative": result.get("ATTACK NARRATIVE", "").strip(),
        "mitre_raw": result.get("MITRE ATT&CK MAPPING", "").strip(),
        "recommended_actions": actions,
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def generate_placeholder_report(alert_id: str, anomalous_edges: int) -> ReportArtifact:
    """Deterministic fallback when no LLM is available."""
    return ReportArtifact(
        verdict="suspicious",
        confidence="low",
        narrative=(
            f"Alert {alert_id} contains {anomalous_edges} anomalous aggregated "
            "edge(s) whose behaviour deviated significantly from the benign training "
            "baseline. Manual analyst review is recommended."
        ),
        note="LLM explainer not configured. Set OLLAMA_BASE_URL or OPENAI_API_KEY to enable.",
    )


def generate_report(
    alert_payload: dict,
    config: LLMConfig | None = None,
) -> ReportArtifact:
    """
    Generate an investigation report for the given alert payload.

    Parameters
    ----------
    alert_payload:
        The serialized alert dict produced by ``serialize_alert_subgraph`` or
        ``serialize_reduced_subgraph``.
    config:
        LLMConfig to use. If None, ``config_from_env()`` is called automatically.
        If no provider is reachable, falls back to a placeholder report.
    """
    if config is None:
        config = config_from_env()

    alert_id = str(alert_payload.get("alert_id", "unknown"))
    anomalous_edges = int(alert_payload.get("flagged_edge_count", 0))

    if config is None:
        return generate_placeholder_report(alert_id, anomalous_edges)

    # Attach MITRE pre-mapping derived from the graph edges so the LLM has
    # structured hints even if it doesn't know the full ATT&CK catalogue.
    edge_hints = [
        {
            "src_type": node_lookup(alert_payload, e.get("src", "")),
            "edge_type": e.get("type", ""),
            "dst_type": node_lookup(alert_payload, e.get("dst", "")),
        }
        for e in alert_payload.get("edges", [])
    ]
    alert_payload = dict(alert_payload)
    alert_payload["mitre_hints"] = map_subgraph(edge_hints)

    prompt = _build_user_message(alert_payload)

    try:
        if config.provider == "openai":
            raw_text = _call_openai(prompt, config)
        else:
            raw_text = _call_ollama(prompt, config)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, OSError) as exc:
        return ReportArtifact(
            verdict="suspicious",
            confidence="low",
            narrative=(
                f"LLM call failed ({exc}). Alert {alert_id} has {anomalous_edges} "
                "anomalous edges — manual review required."
            ),
            note=f"LLM error: {exc}",
        )

    parsed = _parse_response(raw_text)

    # Merge rule-based MITRE with any the LLM added
    mitre = list(alert_payload.get("mitre_hints", []))
    seen_ids = {m["technique_id"] for m in mitre}
    for line in parsed["mitre_raw"].splitlines():
        line = line.strip().lstrip("-•* ")
        if not line:
            continue
        parts = line.split(" - ", 1)
        tid = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else line
        if tid not in seen_ids:
            mitre.append({"technique_id": tid, "technique_name": name,
                           "tactic": "", "rationale": "LLM-identified"})
            seen_ids.add(tid)

    return ReportArtifact(
        verdict=parsed["verdict"],
        confidence=parsed["confidence"],
        narrative=parsed["narrative"],
        mitre_techniques=mitre,
        recommended_actions=parsed["recommended_actions"],
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def node_lookup(alert_payload: dict, node_id: str) -> str:
    """Return the node_type for a given node_id from an alert payload."""
    for node in alert_payload.get("nodes", []):
        if node.get("id") == node_id:
            return node.get("type", "")
    return ""
