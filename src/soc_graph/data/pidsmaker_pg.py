from __future__ import annotations

"""
PIDSMaker PostgreSQL adapter.

PIDSMaker (https://github.com/ubc-provenance/PIDSMaker) ships pre-processed
PostgreSQL dumps of all DARPA TC datasets.  This module connects to the
database and exports a provenance graph in our internal format.

Schema
------
PIDSMaker normalises every dataset into two tables:

    node(id BIGINT, type VARCHAR, name VARCHAR)
    edge(id BIGINT, type VARCHAR, src BIGINT, dst BIGINT, ts BIGINT)

Column meanings
    node.type  : 'process' | 'file' | 'netflow'
    edge.type  : see _PG_EDGE_MAP below
    edge.ts    : Unix timestamp in nanoseconds (or seconds — detected automatically)

Usage
-----
    from soc_graph.data.pidsmaker_pg import export_from_postgres

    events = export_from_postgres(
        dsn="postgresql://user:pass@localhost:5432/cadets_e3",
        dataset="cadets",       # used for progress messages only
        limit=None,             # set an int to cap during exploration
    )

Or export to CSV for offline use:

    python scripts/export_pidsmaker_pg.py \
        --dsn "postgresql://user:pass@localhost:5432/cadets_e3" \
        --out data/processed/cadets_e3.csv

Requires: psycopg2  (pip install psycopg2-binary)
"""

import logging
from datetime import datetime, timezone
from typing import Iterator

from .schemas import EdgeType, Node, NodeType, ProvenanceEvent

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type maps
# ---------------------------------------------------------------------------

_PG_NODE_MAP: dict[str, NodeType] = {
    "process": NodeType.PROCESS,
    "file":    NodeType.FILE,
    "netflow": NodeType.SOCKET,
    # aliases seen in some versions
    "net_flow_object": NodeType.SOCKET,
    "subject":         NodeType.PROCESS,
    "file_object":     NodeType.FILE,
}

_PG_EDGE_MAP: dict[str, EdgeType] = {
    # PIDSMaker canonical names
    "read":       EdgeType.READ,
    "write":      EdgeType.WRITE,
    "execute":    EdgeType.EXECUTE,
    "fork":       EdgeType.FORK,
    "clone":      EdgeType.FORK,
    "connect":    EdgeType.CONNECT,
    "accept":     EdgeType.CONNECT,
    "send":       EdgeType.SEND,
    "sendto":     EdgeType.SEND,
    "sendmsg":    EdgeType.SEND,
    "recv":       EdgeType.RECV,
    "recvfrom":   EdgeType.RECV,
    "recvmsg":    EdgeType.RECV,
    # CDM event names (some PIDSMaker versions keep these)
    "event_read":     EdgeType.READ,
    "event_write":    EdgeType.WRITE,
    "event_execute":  EdgeType.EXECUTE,
    "event_fork":     EdgeType.FORK,
    "event_clone":    EdgeType.FORK,
    "event_connect":  EdgeType.CONNECT,
    "event_sendto":   EdgeType.SEND,
    "event_sendmsg":  EdgeType.SEND,
    "event_recvfrom": EdgeType.RECV,
    "event_recvmsg":  EdgeType.RECV,
    "event_mmap":     EdgeType.READ,
}

# Nanosecond threshold: timestamps > 1e15 are almost certainly nanoseconds
_NS_THRESHOLD = 1_000_000_000_000_000


def _ts_to_dt(ts: int) -> datetime:
    if ts > _NS_THRESHOLD:
        ts = ts // 1_000_000_000
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Core export
# ---------------------------------------------------------------------------

def _connect(dsn: str):
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for the PostgreSQL ingestion path.\n"
            "Install it with:  pip install psycopg2-binary"
        ) from exc
    return psycopg2.connect(dsn)


def export_from_postgres(
    dsn: str,
    dataset: str = "dataset",
    node_table: str = "node",
    edge_table: str = "edge",
    limit: int | None = None,
    batch_size: int = 50_000,
) -> list[ProvenanceEvent]:
    """
    Pull the full provenance graph from a PIDSMaker PostgreSQL database.

    Parameters
    ----------
    dsn        : libpq connection string, e.g. "postgresql://u:p@host/db"
    dataset    : human label for log messages
    node_table : name of the node table (default "node")
    edge_table : name of the edge table (default "edge")
    limit      : cap the number of edges to read (useful for exploration)
    batch_size : rows per cursor fetch

    Returns
    -------
    list[ProvenanceEvent], sorted by timestamp ascending.
    """
    return list(stream_from_postgres(
        dsn=dsn, dataset=dataset,
        node_table=node_table, edge_table=edge_table,
        limit=limit, batch_size=batch_size,
    ))


def stream_from_postgres(
    dsn: str,
    dataset: str = "dataset",
    node_table: str = "node",
    edge_table: str = "edge",
    limit: int | None = None,
    batch_size: int = 50_000,
) -> Iterator[ProvenanceEvent]:
    """Streaming variant — lower peak memory for very large datasets."""
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            # Build node lookup
            log.info("[%s] Loading node table …", dataset)
            cur.execute(f"SELECT id, type, name FROM {node_table}")  # noqa: S608
            nodes: dict[int, Node] = {}
            for row in cur.fetchall():
                nid, ntype_raw, name = row
                ntype = _PG_NODE_MAP.get(str(ntype_raw).lower())
                if ntype is None:
                    continue
                nodes[int(nid)] = Node(
                    node_id=str(nid),
                    node_type=ntype,
                    name=str(name) if name else f"node:{nid}",
                )
            log.info("[%s] %d nodes loaded", dataset, len(nodes))

            # Stream edges
            limit_clause = f"LIMIT {limit}" if limit else ""
            cur.execute(
                f"SELECT id, type, src, dst, ts FROM {edge_table} ORDER BY ts {limit_clause}"  # noqa: S608
            )
            emitted = skipped = 0

            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                for eid, etype_raw, src_id, dst_id, ts_raw in rows:
                    edge_type = _PG_EDGE_MAP.get(str(etype_raw).lower())
                    if edge_type is None:
                        skipped += 1
                        continue
                    src = nodes.get(int(src_id))
                    dst = nodes.get(int(dst_id))
                    if src is None or dst is None:
                        skipped += 1
                        continue
                    try:
                        ts = _ts_to_dt(int(ts_raw))
                    except (ValueError, OSError):
                        skipped += 1
                        continue
                    yield ProvenanceEvent(
                        event_id=str(eid),
                        timestamp=ts,
                        source=src,
                        target=dst,
                        edge_type=edge_type,
                        raw_event_type=str(etype_raw),
                    )
                    emitted += 1

            log.info(
                "[%s] %d events emitted, %d skipped",
                dataset, emitted, skipped,
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CSV export helper
# ---------------------------------------------------------------------------

def export_to_csv(events: list[ProvenanceEvent], out_path: str) -> int:
    """
    Write a list of ProvenanceEvents to the PIDSMaker-style CSV format
    that soc_graph.data.pidsmaker.load_events() can read directly.

    Returns the number of rows written.
    """
    import csv
    from pathlib import Path

    fieldnames = [
        "event_id", "timestamp", "edge_type",
        "src_id", "src_type", "src_name",
        "dst_id", "dst_type", "dst_name",
        "raw_event_type",
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow({
                "event_id":      ev.event_id,
                "timestamp":     ev.timestamp.isoformat().replace("+00:00", "Z"),
                "edge_type":     ev.edge_type.value,
                "src_id":        ev.source.node_id,
                "src_type":      ev.source.node_type.value,
                "src_name":      ev.source.name,
                "dst_id":        ev.target.node_id,
                "dst_type":      ev.target.node_type.value,
                "dst_name":      ev.target.name,
                "raw_event_type": ev.raw_event_type or "",
            })
    return len(events)
