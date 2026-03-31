from datetime import datetime, timezone
from pathlib import Path

from soc_graph.data.pidsmaker import load_records, records_from_rows


def test_records_from_rows_validates_and_normalizes_timestamps() -> None:
    records = records_from_rows(
        [
            {
                "event_id": "evt-1",
                "timestamp": "2026-01-01T12:00:00Z",
                "edge_type": "WRITE",
                "src_id": "proc-1",
                "src_type": "PROCESS",
                "src_name": "/bin/bash",
                "dst_id": "file-1",
                "dst_type": "FILE",
                "dst_name": "/tmp/out",
            }
        ]
    )

    assert len(records) == 1
    assert records[0].timestamp == datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_load_records_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        (
            "event_id,timestamp,edge_type,src_id,src_type,src_name,dst_id,dst_type,dst_name\n"
            "evt-1,2026-01-01T12:00:00Z,WRITE,proc-1,PROCESS,/bin/bash,file-1,FILE,/tmp/out\n"
        ),
        encoding="utf-8",
    )

    records = load_records(csv_path)

    assert len(records) == 1
    assert records[0].event_id == "evt-1"
