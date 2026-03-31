from datetime import datetime, timezone

from soc_graph.data.pidsmaker import PIDSMakerRecord, normalize_record
from soc_graph.data.schemas import EdgeType, NodeType


def test_normalize_record_maps_types() -> None:
    record = PIDSMakerRecord(
        event_id="evt-1",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        edge_type="write",
        src_id="proc-1",
        src_type="process",
        src_name="/bin/bash",
        dst_id="file-1",
        dst_type="file",
        dst_name="/tmp/out",
    )

    event = normalize_record(record)

    assert event.edge_type is EdgeType.WRITE
    assert event.source.node_type is NodeType.PROCESS
    assert event.target.node_type is NodeType.FILE

