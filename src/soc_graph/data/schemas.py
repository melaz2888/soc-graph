from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class NodeType(str, Enum):
    PROCESS = "PROCESS"
    FILE = "FILE"
    SOCKET = "SOCKET"


class EdgeType(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    EXECUTE = "EXECUTE"
    CONNECT = "CONNECT"
    SEND = "SEND"
    RECV = "RECV"
    FORK = "FORK"


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: NodeType
    name: str


@dataclass(frozen=True)
class ProvenanceEvent:
    event_id: str
    timestamp: datetime
    source: Node
    target: Node
    edge_type: EdgeType
    actor_process_id: str | None = None
    raw_event_type: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregatedEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    count: int
    first_seen: datetime
    last_seen: datetime


@dataclass
class GraphSnapshot:
    window_start: datetime
    window_end: datetime
    nodes: dict[str, Node]
    edges: list[AggregatedEdge]

    @property
    def node_type_counts(self) -> Counter[NodeType]:
        return Counter(node.node_type for node in self.nodes.values())

    @property
    def edge_type_counts(self) -> Counter[EdgeType]:
        return Counter(edge.edge_type for edge in self.edges)

    @property
    def total_edge_observations(self) -> int:
        return sum(edge.count for edge in self.edges)


@dataclass(frozen=True)
class GraphTensorArtifact:
    """
    Lightweight graph artifact that mirrors the future PyG payload shape.

    This lets us formalize graph features now without forcing a hard dependency
    on torch or torch-geometric in the first implementation milestone.
    """

    window_start: datetime
    window_end: datetime
    node_ids: list[str]
    edge_keys: list[str]
    node_features: list[list[float]]
    edge_index: list[list[int]]
    edge_features: list[list[float]]
    edge_counts: list[float]

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return len(self.edge_keys)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_time(dt: datetime, window: timedelta) -> datetime:
    dt = ensure_utc(dt)
    seconds = int(window.total_seconds())
    if seconds <= 0:
        raise ValueError("window must be positive")
    floored = int(dt.timestamp()) // seconds * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)
