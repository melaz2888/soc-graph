from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .schemas import EdgeType, Node, NodeType, ProvenanceEvent, ensure_utc


@dataclass(frozen=True)
class PIDSMakerRecord:
    event_id: str
    timestamp: datetime
    edge_type: str
    src_id: str
    src_type: str
    src_name: str
    dst_id: str
    dst_type: str
    dst_name: str
    actor_process_id: str | None = None
    raw_event_type: str | None = None


REQUIRED_COLUMNS = (
    "event_id",
    "timestamp",
    "edge_type",
    "src_id",
    "src_type",
    "src_name",
    "dst_id",
    "dst_type",
    "dst_name",
)

EDGE_TYPE_MAP: dict[str, EdgeType] = {
    "READ": EdgeType.READ,
    "WRITE": EdgeType.WRITE,
    "EXECUTE": EdgeType.EXECUTE,
    "CONNECT": EdgeType.CONNECT,
    "SEND": EdgeType.SEND,
    "RECV": EdgeType.RECV,
    "FORK": EdgeType.FORK,
}

NODE_TYPE_MAP: dict[str, NodeType] = {
    "PROCESS": NodeType.PROCESS,
    "FILE": NodeType.FILE,
    "SOCKET": NodeType.SOCKET,
}


def normalize_record(record: PIDSMakerRecord) -> ProvenanceEvent:
    try:
        edge_type = EDGE_TYPE_MAP[record.edge_type.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported edge type: {record.edge_type}") from exc

    try:
        src_type = NODE_TYPE_MAP[record.src_type.upper()]
        dst_type = NODE_TYPE_MAP[record.dst_type.upper()]
    except KeyError as exc:
        raise ValueError("unsupported node type in PIDSMaker record") from exc

    return ProvenanceEvent(
        event_id=record.event_id,
        timestamp=ensure_utc(record.timestamp),
        source=Node(node_id=record.src_id, node_type=src_type, name=record.src_name),
        target=Node(node_id=record.dst_id, node_type=dst_type, name=record.dst_name),
        edge_type=edge_type,
        actor_process_id=record.actor_process_id,
        raw_event_type=record.raw_event_type,
    )


def normalize_records(records: Iterable[PIDSMakerRecord]) -> list[ProvenanceEvent]:
    return [normalize_record(record) for record in records]


def _missing_columns(column_names: Iterable[str]) -> list[str]:
    names = set(column_names)
    return [column for column in REQUIRED_COLUMNS if column not in names]


def records_from_rows(rows: Iterable[dict[str, str]]) -> list[PIDSMakerRecord]:
    materialized = list(rows)
    missing = _missing_columns(materialized[0].keys() if materialized else REQUIRED_COLUMNS)
    if missing:
        raise ValueError(f"missing PIDSMaker columns: {', '.join(missing)}")

    return [
        PIDSMakerRecord(
            event_id=str(row["event_id"]),
            timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
            edge_type=str(row["edge_type"]),
            src_id=str(row["src_id"]),
            src_type=str(row["src_type"]),
            src_name=str(row["src_name"]),
            dst_id=str(row["dst_id"]),
            dst_type=str(row["dst_type"]),
            dst_name=str(row["dst_name"]),
            actor_process_id=str(row["actor_process_id"]) if row.get("actor_process_id") else None,
            raw_event_type=str(row["raw_event_type"]) if row.get("raw_event_type") else None,
        )
        for row in materialized
    ]


def load_records(path: str | Path) -> list[PIDSMakerRecord]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return records_from_rows(csv.DictReader(handle))
    elif suffix == ".parquet":
        raise NotImplementedError(
            "Parquet ingestion requires an optional dataframe engine that is not enabled in this environment yet."
        )
    else:
        raise ValueError("PIDSMaker input must be a .csv or .parquet file")


def load_events(path: str | Path) -> list[ProvenanceEvent]:
    return normalize_records(load_records(path))
