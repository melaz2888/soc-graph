from datetime import datetime, timezone
from pathlib import Path

from soc_graph.data.pidsmaker import load_events
from soc_graph.data.pidsmaker_pg import export_to_csv
from soc_graph.data.schemas import EdgeType, Node, NodeType, ProvenanceEvent


def test_export_to_csv_round_trips_with_csv_loader(tmp_path: Path) -> None:
    proc = Node("proc-1", NodeType.PROCESS, "/usr/bin/python3")
    file_node = Node("file-1", NodeType.FILE, "/tmp/out")
    event = ProvenanceEvent(
        event_id="evt-1",
        timestamp=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        source=proc,
        target=file_node,
        edge_type=EdgeType.WRITE,
        raw_event_type="event_write",
    )

    output = tmp_path / "events.csv"
    rows = export_to_csv([event], str(output))
    loaded = load_events(output)

    assert rows == 1
    assert len(loaded) == 1
    assert loaded[0].edge_type is EdgeType.WRITE
    assert loaded[0].source.name == "/usr/bin/python3"
