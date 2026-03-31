from datetime import datetime, timedelta, timezone

from soc_graph.data.build_graph import build_graph_artifacts, build_snapshot
from soc_graph.data.schemas import EdgeType, Node, NodeType, ProvenanceEvent
from soc_graph.detection.evaluate import evaluate_window_predictions
from soc_graph.model.detector import BehavioralAnomalyDetector
from soc_graph.model.train import detect_anomalies, fit_detector


def _events() -> list[ProvenanceEvent]:
    proc = Node("proc-1", NodeType.PROCESS, "/bin/bash")
    file_node = Node("file-1", NodeType.FILE, "/tmp/out")
    socket = Node("sock-1", NodeType.SOCKET, "10.0.0.1:443")
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return [
        ProvenanceEvent("evt-1", base, proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-2", base + timedelta(minutes=1), proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-3", base + timedelta(minutes=16), proc, socket, EdgeType.CONNECT),
        ProvenanceEvent("evt-4", base + timedelta(minutes=17), proc, socket, EdgeType.CONNECT),
        ProvenanceEvent("evt-5", base + timedelta(minutes=18), proc, socket, EdgeType.CONNECT),
        ProvenanceEvent("evt-6", base + timedelta(minutes=19), proc, socket, EdgeType.CONNECT),
    ]


def test_training_summary_and_window_evaluation() -> None:
    artifacts = build_graph_artifacts(_events(), window=timedelta(minutes=15))
    detector = BehavioralAnomalyDetector()

    summary = fit_detector(detector, artifacts[:1], threshold_k=1.0)
    detections = detect_anomalies(detector, artifacts, threshold=summary.learned_threshold)
    evaluation = evaluate_window_predictions(
        predicted_windows=[bool(window_scores) for window_scores in detections],
        ground_truth_windows=[False, True],
    )

    assert summary.num_windows == 1
    assert summary.learned_threshold > 0
    assert len(detections) == 2
    assert evaluation.recall == 1.0
