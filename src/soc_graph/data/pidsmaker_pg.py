from __future__ import annotations

"""
PIDSMaker PostgreSQL adapter.

Supports both:
- the simplified `node` / `edge` schema used in early scaffolding
- the split CADETS-style schema restored from the real PIDSMaker dumps:
  `event_table`, `subject_node_table`, `file_node_table`, `netflow_node_table`

The split schema is now the canonical path for CADETS E3.
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

from .schemas import EdgeType, Node, NodeType, ProvenanceEvent

log = logging.getLogger(__name__)

_PG_NODE_MAP: dict[str, NodeType] = {
    "process": NodeType.PROCESS,
    "file": NodeType.FILE,
    "netflow": NodeType.SOCKET,
    "net_flow_object": NodeType.SOCKET,
    "subject": NodeType.PROCESS,
    "file_object": NodeType.FILE,
}

_PG_EDGE_MAP: dict[str, EdgeType] = {
    "read": EdgeType.READ,
    "write": EdgeType.WRITE,
    "execute": EdgeType.EXECUTE,
    "fork": EdgeType.FORK,
    "clone": EdgeType.FORK,
    "connect": EdgeType.CONNECT,
    "accept": EdgeType.CONNECT,
    "send": EdgeType.SEND,
    "sendto": EdgeType.SEND,
    "sendmsg": EdgeType.SEND,
    "recv": EdgeType.RECV,
    "recvfrom": EdgeType.RECV,
    "recvmsg": EdgeType.RECV,
    "event_read": EdgeType.READ,
    "event_write": EdgeType.WRITE,
    "event_execute": EdgeType.EXECUTE,
    "event_fork": EdgeType.FORK,
    "event_clone": EdgeType.FORK,
    "event_connect": EdgeType.CONNECT,
    "event_accept": EdgeType.CONNECT,
    "event_sendto": EdgeType.SEND,
    "event_sendmsg": EdgeType.SEND,
    "event_recvfrom": EdgeType.RECV,
    "event_recvmsg": EdgeType.RECV,
    "event_mmap": EdgeType.READ,
}

_NS_THRESHOLD = 1_000_000_000_000_000


def _ts_to_dt(ts: int) -> datetime:
    if ts > _NS_THRESHOLD:
        ts = ts // 1_000_000_000
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _connect(dsn: str):
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError(
            "psycopg2 is required for the PostgreSQL ingestion path.\n"
            "Install it with: pip install psycopg2-binary"
        ) from exc
    return psycopg2.connect(dsn)


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
    return cur.fetchone()[0] is not None


def _resolve_schema(cur, node_table: str, edge_table: str) -> str:
    split_tables = (
        "event_table",
        "subject_node_table",
        "file_node_table",
        "netflow_node_table",
    )
    if all(_table_exists(cur, table) for table in split_tables):
        return "split"
    if _table_exists(cur, node_table) and _table_exists(cur, edge_table):
        return "simple"
    raise ValueError(
        "Could not detect a supported PIDSMaker PostgreSQL schema. "
        f"Tried split tables {split_tables} and simple tables '{node_table}'/'{edge_table}'."
    )


def export_from_postgres(
    dsn: str,
    dataset: str = "dataset",
    node_table: str = "node",
    edge_table: str = "edge",
    limit: int | None = None,
    batch_size: int = 50_000,
) -> list[ProvenanceEvent]:
    return list(
        stream_from_postgres(
            dsn=dsn,
            dataset=dataset,
            node_table=node_table,
            edge_table=edge_table,
            limit=limit,
            batch_size=batch_size,
        )
    )


def stream_from_postgres(
    dsn: str,
    dataset: str = "dataset",
    node_table: str = "node",
    edge_table: str = "edge",
    limit: int | None = None,
    batch_size: int = 50_000,
) -> Iterator[ProvenanceEvent]:
    """Yield normalized provenance events from a PIDSMaker PostgreSQL database."""
    conn = _connect(dsn)
    try:
        with conn.cursor() as cur:
            schema = _resolve_schema(cur, node_table=node_table, edge_table=edge_table)

        if schema == "split":
            yield from _stream_split_schema(
                conn=conn,
                dataset=dataset,
                limit=limit,
                batch_size=batch_size,
            )
            return

        yield from _stream_simple_schema(
            conn=conn,
            dataset=dataset,
            node_table=node_table,
            edge_table=edge_table,
            limit=limit,
            batch_size=batch_size,
        )
    finally:
        conn.close()


def _stream_simple_schema(
    conn,
    dataset: str,
    node_table: str,
    edge_table: str,
    limit: int | None,
    batch_size: int,
) -> Iterator[ProvenanceEvent]:
    with conn.cursor() as cur:
        log.info("[%s] Loading simple node table ...", dataset)
        cur.execute(f"SELECT id, type, name FROM {node_table}")  # noqa: S608
        nodes: dict[int, Node] = {}
        for nid, ntype_raw, name in cur.fetchall():
            ntype = _PG_NODE_MAP.get(str(ntype_raw).lower())
            if ntype is None:
                continue
            nodes[int(nid)] = Node(
                node_id=str(nid),
                node_type=ntype,
                name=str(name) if name else f"node:{nid}",
            )
        log.info("[%s] %d nodes loaded", dataset, len(nodes))

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

    log.info("[%s] %d events emitted, %d skipped", dataset, emitted, skipped)


def _stream_split_schema(
    conn,
    dataset: str,
    limit: int | None,
    batch_size: int,
) -> Iterator[ProvenanceEvent]:
    query = """
        WITH nodes AS (
            SELECT
                hash_id,
                index_id::text AS index_id,
                'process' AS node_type,
                COALESCE(NULLIF(path, ''), NULLIF(cmd, ''), node_uuid) AS node_name
            FROM subject_node_table
            UNION ALL
            SELECT
                hash_id,
                index_id::text AS index_id,
                'file' AS node_type,
                COALESCE(NULLIF(path, ''), node_uuid) AS node_name
            FROM file_node_table
            UNION ALL
            SELECT
                hash_id,
                index_id::text AS index_id,
                'netflow' AS node_type,
                COALESCE(
                    NULLIF(
                        CONCAT_WS(':', NULLIF(src_addr, ''), NULLIF(src_port, ''))
                        || '->' ||
                        CONCAT_WS(':', NULLIF(dst_addr, ''), NULLIF(dst_port, '')),
                        '->'
                    ),
                    node_uuid
                ) AS node_name
            FROM netflow_node_table
        )
        SELECT
            e.event_uuid,
            e.operation,
            e.timestamp_rec,
            e.src_node,
            e.src_index_id,
            src.node_type,
            src.node_name,
            e.dst_node,
            e.dst_index_id,
            dst.node_type,
            dst.node_name
        FROM event_table AS e
        JOIN nodes AS src
          ON src.hash_id = e.src_node
         AND src.index_id = e.src_index_id
        JOIN nodes AS dst
          ON dst.hash_id = e.dst_node
         AND dst.index_id = e.dst_index_id
        ORDER BY e.timestamp_rec
    """
    if limit:
        query += f"\nLIMIT {int(limit)}"

    emitted = skipped = 0
    node_cache: dict[tuple[str, str, str], Node] = {}
    with conn.cursor(name="pidsmaker_event_stream") as cur:
        cur.itersize = batch_size
        log.info("[%s] Streaming events from split PIDSMaker schema ...", dataset)
        cur.execute(query)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for (
                eid,
                etype_raw,
                ts_raw,
                src_hash,
                src_index_id,
                src_type_raw,
                src_name,
                dst_hash,
                dst_index_id,
                dst_type_raw,
                dst_name,
            ) in rows:
                edge_type = _PG_EDGE_MAP.get(str(etype_raw).lower())
                src_type = _PG_NODE_MAP.get(str(src_type_raw).lower())
                dst_type = _PG_NODE_MAP.get(str(dst_type_raw).lower())
                if edge_type is None or src_type is None or dst_type is None:
                    skipped += 1
                    continue
                try:
                    ts = _ts_to_dt(int(ts_raw))
                except (ValueError, OSError):
                    skipped += 1
                    continue

                src_key = (str(src_hash), str(src_index_id), src_type.value)
                dst_key = (str(dst_hash), str(dst_index_id), dst_type.value)

                src = node_cache.get(src_key)
                if src is None:
                    src = Node(
                        node_id=f"{src_hash}:{src_index_id}",
                        node_type=src_type,
                        name=str(src_name) if src_name else f"node:{src_hash}",
                    )
                    node_cache[src_key] = src

                dst = node_cache.get(dst_key)
                if dst is None:
                    dst = Node(
                        node_id=f"{dst_hash}:{dst_index_id}",
                        node_type=dst_type,
                        name=str(dst_name) if dst_name else f"node:{dst_hash}",
                    )
                    node_cache[dst_key] = dst

                yield ProvenanceEvent(
                    event_id=str(eid),
                    timestamp=ts,
                    source=src,
                    target=dst,
                    edge_type=edge_type,
                    raw_event_type=str(etype_raw),
                )
                emitted += 1

    log.info("[%s] %d events emitted, %d skipped", dataset, emitted, skipped)


def _write_events_to_csv(events: Iterable[ProvenanceEvent], out_path: str) -> int:
    fieldnames = [
        "event_id",
        "timestamp",
        "edge_type",
        "src_id",
        "src_type",
        "src_name",
        "dst_id",
        "dst_type",
        "dst_name",
        "raw_event_type",
    ]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in events:
            writer.writerow(
                {
                    "event_id": ev.event_id,
                    "timestamp": ev.timestamp.isoformat().replace("+00:00", "Z"),
                    "edge_type": ev.edge_type.value,
                    "src_id": ev.source.node_id,
                    "src_type": ev.source.node_type.value,
                    "src_name": ev.source.name,
                    "dst_id": ev.target.node_id,
                    "dst_type": ev.target.node_type.value,
                    "dst_name": ev.target.name,
                    "raw_event_type": ev.raw_event_type or "",
                }
            )
            rows += 1
    return rows


def export_to_csv(events: list[ProvenanceEvent], out_path: str) -> int:
    return _write_events_to_csv(events, out_path)


def export_stream_to_csv(events: Iterable[ProvenanceEvent], out_path: str) -> int:
    return _write_events_to_csv(events, out_path)
