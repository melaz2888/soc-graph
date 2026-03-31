from datetime import datetime, timedelta, timezone

from soc_graph.data.build_graph import build_graph_artifacts
from soc_graph.data.schemas import EdgeType, Node, NodeType, ProvenanceEvent


def test_build_graph_artifacts_materializes_expected_shapes() -> None:
    proc = Node("proc-1", NodeType.PROCESS, "/bin/bash")
    file_node = Node("file-1", NodeType.FILE, "/tmp/out")
    socket = Node("sock-1", NodeType.SOCKET, "10.0.0.1:443")
    ts = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

    events = [
        ProvenanceEvent("evt-1", ts, proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-2", ts + timedelta(minutes=2), proc, socket, EdgeType.CONNECT),
    ]

    artifacts = build_graph_artifacts(events, window=timedelta(minutes=15))

    assert len(artifacts) == 1
    assert len(artifacts[0].node_features) == 3
    assert len(artifacts[0].node_features[0]) == 6
    assert len(artifacts[0].edge_index) == 2
    assert len(artifacts[0].edge_index[0]) == 2
    assert len(artifacts[0].edge_features) == 2
    assert len(artifacts[0].edge_features[0]) == 8
