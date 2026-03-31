from __future__ import annotations

from dataclasses import dataclass, field

from soc_graph.data.schemas import AggregatedEdge, GraphSnapshot, Node


@dataclass(frozen=True)
class CandidateAttackSubgraph:
    alert_id: str
    edge_keys: list[str]
    flagged_edge_count: int
    total_edge_count: int


@dataclass(frozen=True)
class ReducedAttackSubgraph:
    """
    Compact attack summary graph produced by reducing flagged edges to their
    connected components.  This is what gets serialised for the LLM and the API.
    """

    alert_id: str
    nodes: dict[str, Node]
    edges: list[AggregatedEdge]
    anomaly_scores: dict[str, float]
    flagged_edge_count: int
    total_edge_count: int
    component_count: int


# ---------------------------------------------------------------------------
# Internal union-find for connected-component extraction (no extra deps)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def _ensure(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: str) -> str:
        self._ensure(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        self._parent[self.find(x)] = self.find(y)

    def components(self, nodes: set[str]) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for n in nodes:
            root = self.find(n)
            groups.setdefault(root, set()).add(n)
        return list(groups.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_candidate_subgraph(
    snapshot: GraphSnapshot,
    flagged_scores: dict[str, float],
    alert_id: str,
) -> CandidateAttackSubgraph:
    """Backward-compatible thin wrapper kept for existing scripts."""
    return CandidateAttackSubgraph(
        alert_id=alert_id,
        edge_keys=sorted(flagged_scores),
        flagged_edge_count=len(flagged_scores),
        total_edge_count=len(snapshot.edges),
    )


def build_reduced_subgraph(
    snapshot: GraphSnapshot,
    flagged_scores: dict[str, float],
    alert_id: str,
) -> ReducedAttackSubgraph:
    """
    Build a compact attack summary graph from flagged edges.

    Steps:
    1. Collect only the flagged AggregatedEdge objects.
    2. Gather the node IDs referenced by those edges.
    3. Run union-find to count connected components.
    4. Return a ReducedAttackSubgraph with trimmed node/edge sets.
    """
    flagged_key_set = set(flagged_scores)

    flagged_edges: list[AggregatedEdge] = [
        edge
        for edge in snapshot.edges
        if f"{edge.source_id}:{edge.edge_type.value}:{edge.target_id}" in flagged_key_set
    ]

    involved_node_ids: set[str] = set()
    uf = _UnionFind()
    for edge in flagged_edges:
        involved_node_ids.add(edge.source_id)
        involved_node_ids.add(edge.target_id)
        uf.union(edge.source_id, edge.target_id)

    components = uf.components(involved_node_ids)

    trimmed_nodes: dict[str, Node] = {
        node_id: node
        for node_id, node in snapshot.nodes.items()
        if node_id in involved_node_ids
    }

    return ReducedAttackSubgraph(
        alert_id=alert_id,
        nodes=trimmed_nodes,
        edges=flagged_edges,
        anomaly_scores=flagged_scores,
        flagged_edge_count=len(flagged_edges),
        total_edge_count=len(snapshot.edges),
        component_count=len(components),
    )
