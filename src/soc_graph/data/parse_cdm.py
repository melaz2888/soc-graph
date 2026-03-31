from __future__ import annotations

"""
DARPA Transparent Computing CDM JSON parser.

Supports CDM v17, v18, v19 (all use the same structure; namespace prefix
differs but the logic is identical).

File formats accepted
---------------------
- Plain newline-delimited JSON  (.json)
- Gzipped newline-delimited JSON (.json.gz)
- A directory containing any mix of the above

Algorithm
---------
Pass 1 — build a UUID-to-Node table by scanning every Subject,
         FileObject and NetFlowObject record.
Pass 2 — stream Event records, resolve src/dst UUIDs, emit
         ProvenanceEvent objects.

Both passes are streaming so arbitrarily large files use O(nodes) memory,
not O(events) memory.

Usage
-----
    from soc_graph.data.parse_cdm import parse_cdm_json

    events = parse_cdm_json("data/raw/cadets_e3/")        # directory
    events = parse_cdm_json("data/raw/ta1-cadets-e3.json.gz")  # single file
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator

from .schemas import EdgeType, Node, NodeType, ProvenanceEvent, ensure_utc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CDM event-type → EdgeType mapping
# ---------------------------------------------------------------------------

_CDM_EDGE_MAP: dict[str, EdgeType] = {
    # reads
    "EVENT_READ": EdgeType.READ,
    "EVENT_MMAP": EdgeType.READ,          # memory-mapped read treated as read
    # writes
    "EVENT_WRITE": EdgeType.WRITE,
    "EVENT_TRUNCATE": EdgeType.WRITE,
    "EVENT_RENAME": EdgeType.WRITE,
    "EVENT_CHMOD": EdgeType.WRITE,
    "EVENT_CHOWN": EdgeType.WRITE,
    "EVENT_LINK": EdgeType.WRITE,
    "EVENT_UNLINK": EdgeType.WRITE,
    "EVENT_SETUID": EdgeType.WRITE,
    # execute
    "EVENT_EXECUTE": EdgeType.EXECUTE,
    "EVENT_LOADLIBRARY": EdgeType.EXECUTE,
    # fork / spawn
    "EVENT_FORK": EdgeType.FORK,
    "EVENT_CLONE": EdgeType.FORK,
    "EVENT_VFORK": EdgeType.FORK,
    # network
    "EVENT_CONNECT": EdgeType.CONNECT,
    "EVENT_ACCEPT": EdgeType.CONNECT,
    "EVENT_BIND": EdgeType.CONNECT,
    "EVENT_SENDTO": EdgeType.SEND,
    "EVENT_SENDMSG": EdgeType.SEND,
    "EVENT_WRITE_SOCKET": EdgeType.SEND,
    "EVENT_RECVFROM": EdgeType.RECV,
    "EVENT_RECVMSG": EdgeType.RECV,
    "EVENT_READ_SOCKET_PARAMS": EdgeType.RECV,
}

# CDM node type prefixes → NodeType
_SUBJECT_TYPES = {"SUBJECT_PROCESS", "SUBJECT_UNIT", "SUBJECT_BASIC_BLOCK"}
_FILE_TYPES = {
    "FILE_OBJECT_FILE", "FILE_OBJECT_BLOCK", "FILE_OBJECT_CHAR",
    "FILE_OBJECT_DIR", "FILE_OBJECT_LINK", "FILE_OBJECT_PIPE",
    "FILE_OBJECT_UNIX_SOCKET",
}

# ---------------------------------------------------------------------------
# CDM namespace helper
# ---------------------------------------------------------------------------

def _unwrap_datum(record: dict) -> tuple[str, dict] | None:
    """
    Return (record_kind, payload) from a top-level CDM record dict.

    Handles both variants:
      {"datum": {"com.bbn.tc.schema.avro.cdm18.Event": {...}}}
      {"TCCDMDatum": {"datum": {...}}}            (older format)
    """
    datum = record.get("datum") or record.get("TCCDMDatum", {}).get("datum", {})
    if not isinstance(datum, dict) or not datum:
        return None
    kind_key = next(iter(datum))
    # Strip namespace prefix  e.g. "com.bbn.tc.schema.avro.cdm18.Event" -> "Event"
    short = kind_key.rsplit(".", 1)[-1]
    return short, datum[kind_key]


def _uuid(value: object) -> str | None:
    """Extract UUID string from either a plain string or a union dict."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # {"com.bbn.tc.schema.avro.cdm18.UUID": "abc..."} or {"string": "abc..."}
        v = next(iter(value.values()), None)
        return str(v) if v else None
    return None


def _str_prop(props: object, *keys: str) -> str:
    """
    Pull a string from a CDM properties map.

    CDM represents properties as:
      {"map": {"path": "/etc/passwd", ...}}     (Avro map encoding)
      {"path": "/etc/passwd", ...}              (flat)
    """
    if not props:
        return ""
    if isinstance(props, dict):
        m = props.get("map", props)
        if isinstance(m, dict):
            for k in keys:
                v = m.get(k)
                if v:
                    return str(v)
    return ""


# ---------------------------------------------------------------------------
# Line streaming helper (plain + gzip)
# ---------------------------------------------------------------------------

def _iter_lines(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
    else:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            yield from f


def _candidate_files(source: Path) -> list[Path]:
    if source.is_dir():
        files = sorted(
            p for p in source.rglob("*")
            if p.suffix in (".json", ".gz") and p.is_file()
        )
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"No .json or .json.gz files found in {source}")
    return files


# ---------------------------------------------------------------------------
# Pass 1 — build UUID → Node table
# ---------------------------------------------------------------------------

def _build_node_table(files: list[Path]) -> dict[str, Node]:
    """Scan all files once and return {uuid: Node} for every subject/object."""
    nodes: dict[str, Node] = {}
    total_files = len(files)

    for fi, path in enumerate(files, 1):
        log.debug("Pass 1 [%d/%d] %s", fi, total_files, path.name)
        for raw_line in _iter_lines(path):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = _unwrap_datum(record)
            if result is None:
                continue
            kind, payload = result

            uuid = _uuid(payload.get("uuid"))
            if not uuid:
                continue

            if kind == "Subject":
                stype = payload.get("type", "")
                if stype not in _SUBJECT_TYPES:
                    continue
                props = payload.get("properties")
                name = (
                    _str_prop(props, "name", "path", "cmdLine")
                    or _str_prop(payload, "name", "path")
                    or f"process:{payload.get('cid', uuid[:8])}"
                )
                nodes[uuid] = Node(node_id=uuid, node_type=NodeType.PROCESS, name=name)

            elif kind == "FileObject":
                ftype = payload.get("type", "")
                if ftype not in _FILE_TYPES:
                    continue
                props = payload.get("properties")
                name = (
                    _str_prop(props, "path", "filename", "name")
                    or _str_prop(payload, "path", "filename")
                    or f"file:{uuid[:8]}"
                )
                nodes[uuid] = Node(node_id=uuid, node_type=NodeType.FILE, name=name)

            elif kind == "NetFlowObject":
                remote_addr = payload.get("remoteAddress") or ""
                remote_port = payload.get("remotePort") or ""
                local_addr = payload.get("localAddress") or ""
                local_port = payload.get("localPort") or ""
                # Prefer remote endpoint; fall back to local
                if remote_addr:
                    name = f"{remote_addr}:{remote_port}"
                elif local_addr:
                    name = f"{local_addr}:{local_port}"
                else:
                    name = f"socket:{uuid[:8]}"
                nodes[uuid] = Node(node_id=uuid, node_type=NodeType.SOCKET, name=name)

            elif kind == "UnnamedPipeObject":
                nodes[uuid] = Node(node_id=uuid, node_type=NodeType.FILE, name=f"pipe:{uuid[:8]}")

    log.info("Pass 1 complete: %d nodes resolved", len(nodes))
    return nodes


# ---------------------------------------------------------------------------
# Pass 2 — emit ProvenanceEvent objects
# ---------------------------------------------------------------------------

def _iter_events(
    files: list[Path],
    nodes: dict[str, Node],
    skip_unknown_types: bool,
) -> Iterator[ProvenanceEvent]:
    skipped_type = 0
    skipped_uuid = 0
    emitted = 0
    total_files = len(files)

    for fi, path in enumerate(files, 1):
        log.debug("Pass 2 [%d/%d] %s", fi, total_files, path.name)
        for raw_line in _iter_lines(path):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            result = _unwrap_datum(record)
            if result is None or result[0] != "Event":
                continue
            _, payload = result

            # Timestamp: nanoseconds → seconds → UTC datetime
            ts_nanos = payload.get("timestampNanos")
            if not ts_nanos:
                continue
            try:
                from datetime import datetime, timezone
                ts = datetime.fromtimestamp(int(ts_nanos) / 1e9, tz=timezone.utc)
            except (ValueError, OSError):
                continue

            # Event type
            cdm_type = payload.get("type", "")
            edge_type = _CDM_EDGE_MAP.get(cdm_type)
            if edge_type is None:
                skipped_type += 1
                if not skip_unknown_types:
                    log.debug("Skipping unmapped event type: %s", cdm_type)
                continue

            # Resolve subject (process that caused the event)
            subj_uuid = _uuid(payload.get("subject"))
            if not subj_uuid or subj_uuid not in nodes:
                skipped_uuid += 1
                continue
            source_node = nodes[subj_uuid]

            # Resolve predicateObject (target: file/socket/process)
            obj_uuid = _uuid(payload.get("predicateObject"))
            if not obj_uuid or obj_uuid not in nodes:
                skipped_uuid += 1
                continue
            target_node = nodes[obj_uuid]

            # For FORK events the subject IS the parent process;
            # predicateObject is the child process.  Direction is already correct.

            event_id = _uuid(payload.get("uuid")) or f"evt-{emitted}"
            yield ProvenanceEvent(
                event_id=event_id,
                timestamp=ts,
                source=source_node,
                target=target_node,
                edge_type=edge_type,
                raw_event_type=cdm_type,
            )
            emitted += 1

    log.info(
        "Pass 2 complete: %d events emitted, %d skipped (unknown type), %d skipped (unresolved UUID)",
        emitted, skipped_type, skipped_uuid,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cdm_json(
    source: str | Path,
    skip_unknown_types: bool = True,
) -> list[ProvenanceEvent]:
    """
    Parse DARPA TC CDM JSON files into ProvenanceEvent objects.

    Parameters
    ----------
    source : str | Path
        Path to a single `.json` or `.json.gz` file, OR a directory
        containing any number of such files (processed in sorted order).
    skip_unknown_types : bool
        If True (default), CDM event types not in the mapping table are
        silently ignored.  If False, a debug log is emitted for each.

    Returns
    -------
    list[ProvenanceEvent] — sorted by timestamp ascending.
    """
    source = Path(source)
    files = _candidate_files(source)
    log.info("CDM parser: %d file(s) to process from %s", len(files), source)

    nodes = _build_node_table(files)
    events = list(_iter_events(files, nodes, skip_unknown_types))
    events.sort(key=lambda e: e.timestamp)
    return events


def stream_cdm_json(
    source: str | Path,
    skip_unknown_types: bool = True,
) -> Iterator[ProvenanceEvent]:
    """
    Streaming variant — yields ProvenanceEvent objects without loading all
    events into memory.  Events are NOT sorted (they come in file order).
    Use this for very large datasets where memory is a concern.
    """
    source = Path(source)
    files = _candidate_files(source)
    nodes = _build_node_table(files)
    yield from _iter_events(files, nodes, skip_unknown_types)
