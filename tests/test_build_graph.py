from datetime import datetime, timedelta, timezone

from soc_graph.data.build_graph import build_snapshot
from soc_graph.data.schemas import EdgeType, Node, NodeType, ProvenanceEvent


def test_build_snapshot_aggregates_identical_edges_within_window() -> None:
    proc = Node("proc-1", NodeType.PROCESS, "/bin/bash")
    file_node = Node("file-1", NodeType.FILE, "/tmp/out")
    ts = datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc)

    events = [
        ProvenanceEvent("evt-1", ts, proc, file_node, EdgeType.WRITE),
        ProvenanceEvent("evt-2", ts + timedelta(minutes=1), proc, file_node, EdgeType.WRITE),
    ]

    snapshots = build_snapshot(events, window=timedelta(minutes=15))

    assert len(snapshots) == 1
    assert len(snapshots[0].edges) == 1
    assert snapshots[0].edges[0].count == 2

