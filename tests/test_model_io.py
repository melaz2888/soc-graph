from datetime import datetime, timedelta, timezone
from pathlib import Path

from soc_graph.data.build_graph import build_graph_artifacts
from soc_graph.data.schemas import EdgeType, Node, NodeType, ProvenanceEvent
from soc_graph.model.detector import BehavioralAnomalyDetector
from soc_graph.model.io import load_detector, save_detector


def _events() -> list[ProvenanceEvent]:
    proc = Node("proc-1", NodeType.PROCESS, "/bin/bash")
    file_node = Node("file-1", NodeType.FILE, "/tmp/out")
    socket = Node("sock-1", NodeType.SOCKET, "10.0.0.1:443")
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return [
        ProvenanceEvent("evt-1", base, proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-2", base + timedelta(minutes=1), proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-3", base + timedelta(minutes=16), proc, socket, EdgeType.CONNECT),
    ]


def test_behavioral_detector_round_trips_through_json(tmp_path: Path) -> None:
    artifacts = build_graph_artifacts(_events(), window=timedelta(minutes=15))
    detector = BehavioralAnomalyDetector()
    detector.fit(artifacts[:1])

    model_path = tmp_path / "detector.json"
    save_detector(detector, model_path)
    restored = load_detector(model_path)

    assert restored.signature_counts == detector.signature_counts
    assert restored.edge_profiles["proc-1:WRITE:file-1"].mean_count == 2.0
