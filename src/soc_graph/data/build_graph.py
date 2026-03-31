from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from itertools import groupby
from typing import Iterable

from .schemas import AggregatedEdge, EdgeType, GraphSnapshot, GraphTensorArtifact, NodeType, ProvenanceEvent, floor_time


def build_snapshot(events: list[ProvenanceEvent], window: timedelta) -> list[GraphSnapshot]:
    if not events:
        return []

    ordered = sorted(events, key=lambda event: event.timestamp)
    grouped = groupby(ordered, key=lambda event: floor_time(event.timestamp, window))
    snapshots: list[GraphSnapshot] = []

    for window_start, bucket in grouped:
        bucket_list = list(bucket)
        nodes = {}
        edge_buckets: dict[tuple[str, str, str], list[ProvenanceEvent]] = defaultdict(list)

        for event in bucket_list:
            nodes[event.source.node_id] = event.source
            nodes[event.target.node_id] = event.target
            edge_buckets[(event.source.node_id, event.target.node_id, event.edge_type.value)].append(event)

        edges = [
            AggregatedEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=events_for_edge[0].edge_type,
                count=len(events_for_edge),
                first_seen=events_for_edge[0].timestamp,
                last_seen=events_for_edge[-1].timestamp,
            )
            for (source_id, target_id, _), events_for_edge in edge_buckets.items()
        ]

        snapshots.append(
            GraphSnapshot(
                window_start=window_start,
                window_end=window_start + window,
                nodes=nodes,
                edges=sorted(edges, key=lambda edge: (edge.source_id, edge.target_id, edge.edge_type.value)),
            )
        )

    return snapshots


def _stable_name_scalar(name: str) -> float:
    hashed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return (hashed % 10_000) / 10_000.0


def _node_feature_vector(node_id: str, snapshot: GraphSnapshot) -> list[float]:
    node = snapshot.nodes[node_id]
    in_degree = sum(edge.count for edge in snapshot.edges if edge.target_id == node_id)
    out_degree = sum(edge.count for edge in snapshot.edges if edge.source_id == node_id)
    return [
        1.0 if node.node_type is NodeType.PROCESS else 0.0,
        1.0 if node.node_type is NodeType.FILE else 0.0,
        1.0 if node.node_type is NodeType.SOCKET else 0.0,
        float(in_degree),
        float(out_degree),
        _stable_name_scalar(node.name),
    ]


def _edge_feature_vector(edge, window_start, window_seconds: float) -> list[float]:
    midpoint_seconds = (edge.first_seen - window_start).total_seconds()
    normalized_time = 0.0 if window_seconds == 0 else midpoint_seconds / window_seconds
    return [
        1.0 if edge.edge_type is EdgeType.READ else 0.0,
        1.0 if edge.edge_type is EdgeType.WRITE else 0.0,
        1.0 if edge.edge_type is EdgeType.EXECUTE else 0.0,
        1.0 if edge.edge_type is EdgeType.CONNECT else 0.0,
        1.0 if edge.edge_type is EdgeType.SEND else 0.0,
        1.0 if edge.edge_type is EdgeType.RECV else 0.0,
        1.0 if edge.edge_type is EdgeType.FORK else 0.0,
        float(normalized_time),
    ]


def snapshot_to_graph_tensor(snapshot: GraphSnapshot) -> GraphTensorArtifact:
    node_ids = sorted(snapshot.nodes)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_keys = [f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}" for edge in snapshot.edges]
    window_seconds = (snapshot.window_end - snapshot.window_start).total_seconds()

    node_features = [
        [_node_feature_vector(node_id, snapshot) for node_id in node_ids],
    ][0]
    edge_index = [
        [node_index[edge.source_id] for edge in snapshot.edges],
        [node_index[edge.target_id] for edge in snapshot.edges],
    ] if snapshot.edges else [[], []]
    edge_features = [
        [_edge_feature_vector(edge, snapshot.window_start, window_seconds) for edge in snapshot.edges],
    ][0] if snapshot.edges else []
    edge_counts = [float(edge.count) for edge in snapshot.edges]

    return GraphTensorArtifact(
        window_start=snapshot.window_start,
        window_end=snapshot.window_end,
        node_ids=node_ids,
        edge_keys=edge_keys,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        edge_counts=edge_counts,
    )


def build_graph_artifacts(events: Iterable[ProvenanceEvent], window: timedelta) -> list[GraphTensorArtifact]:
    snapshots = build_snapshot(list(events), window)
    return [snapshot_to_graph_tensor(snapshot) for snapshot in snapshots]
