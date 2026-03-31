from __future__ import annotations

"""
Rule-based MITRE ATT&CK technique mapping.

Each rule maps a (src_node_type, edge_type, dst_node_type) triple to one or
more (technique_id, technique_name) pairs.  The lookup is intentionally broad
so that even a single flagged edge surfaces relevant techniques.

Reference: https://attack.mitre.org/
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MitreMatch:
    technique_id: str
    technique_name: str
    tactic: str
    rationale: str


# (src_type, edge_type, dst_type) -> list[MitreMatch]
_RULES: dict[tuple[str, str, str], list[MitreMatch]] = {
    # --- Process spawning / execution ---
    ("PROCESS", "FORK", "PROCESS"): [
        MitreMatch("T1059", "Command and Scripting Interpreter", "Execution",
                   "Process spawned a child process via fork/exec."),
        MitreMatch("T1106", "Native API", "Execution",
                   "Direct syscall used to create a new process."),
    ],
    ("PROCESS", "EXECUTE", "FILE"): [
        MitreMatch("T1059", "Command and Scripting Interpreter", "Execution",
                   "Process executed a file directly."),
    ],
    ("FILE", "EXECUTE", "PROCESS"): [
        MitreMatch("T1204", "User Execution", "Execution",
                   "A file was executed, spawning a process."),
        MitreMatch("T1059", "Command and Scripting Interpreter", "Execution",
                   "Script or binary executed from file system."),
    ],
    # --- File write (staging / persistence / defense evasion) ---
    ("PROCESS", "WRITE", "FILE"): [
        MitreMatch("T1074", "Data Staged", "Collection",
                   "Process wrote data to a file, possibly staging for exfiltration."),
        MitreMatch("T1027", "Obfuscated Files or Information", "Defense Evasion",
                   "Written file may be an obfuscated payload or dropped tool."),
        MitreMatch("T1547", "Boot or Logon Autostart Execution", "Persistence",
                   "Write to startup/config location could establish persistence."),
    ],
    # --- File read (discovery / collection) ---
    ("PROCESS", "READ", "FILE"): [
        MitreMatch("T1083", "File and Directory Discovery", "Discovery",
                   "Process read a file, possibly enumerating the file system."),
        MitreMatch("T1005", "Data from Local System", "Collection",
                   "Process collected data from a local file."),
    ],
    # --- Network connections (C2 / exfiltration / lateral movement) ---
    ("PROCESS", "CONNECT", "SOCKET"): [
        MitreMatch("T1071", "Application Layer Protocol", "Command and Control",
                   "Process opened a network socket, possibly for C2 comms."),
        MitreMatch("T1571", "Non-Standard Port", "Command and Control",
                   "Connection to a non-standard port may indicate C2 tunnelling."),
        MitreMatch("T1021", "Remote Services", "Lateral Movement",
                   "Connection could represent lateral movement via remote service."),
    ],
    # --- Data sent out (exfiltration) ---
    ("PROCESS", "SEND", "SOCKET"): [
        MitreMatch("T1048", "Exfiltration Over Alternative Protocol", "Exfiltration",
                   "Process sent data over the network; may be exfiltrating."),
        MitreMatch("T1041", "Exfiltration Over C2 Channel", "Exfiltration",
                   "Data may have been sent back over an established C2 channel."),
    ],
    # --- Data received (tool drop / C2 tasking) ---
    ("SOCKET", "RECV", "PROCESS"): [
        MitreMatch("T1105", "Ingress Tool Transfer", "Command and Control",
                   "Process received data from the network; may be a tool drop."),
        MitreMatch("T1071", "Application Layer Protocol", "Command and Control",
                   "Incoming network data may carry C2 instructions."),
    ],
}

# Wildcard rules applied when edge type alone is distinctive enough
_EDGE_WILDCARD: dict[str, list[MitreMatch]] = {
    "SEND": [
        MitreMatch("T1048", "Exfiltration Over Alternative Protocol", "Exfiltration",
                   "Any outbound data transfer is a candidate for exfiltration."),
    ],
    "CONNECT": [
        MitreMatch("T1071", "Application Layer Protocol", "Command and Control",
                   "Any new network connection warrants C2 investigation."),
    ],
}


def lookup(src_type: str, edge_type: str, dst_type: str) -> list[MitreMatch]:
    """Return ATT&CK matches for an (src, edge, dst) triple."""
    key = (src_type.upper(), edge_type.upper(), dst_type.upper())
    specific = _RULES.get(key, [])
    wildcard = _EDGE_WILDCARD.get(edge_type.upper(), [])
    seen_ids: set[str] = {m.technique_id for m in specific}
    combined = list(specific)
    for match in wildcard:
        if match.technique_id not in seen_ids:
            combined.append(match)
            seen_ids.add(match.technique_id)
    return combined


def map_subgraph(
    edges: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Map a list of edge dicts (each with 'src_type', 'edge_type', 'dst_type')
    to a deduplicated list of ATT&CK technique dicts.

    Returns a list of dicts suitable for JSON serialization.
    """
    seen_ids: set[str] = set()
    results: list[dict[str, str]] = []
    for edge in edges:
        src = edge.get("src_type", "")
        etype = edge.get("edge_type", "")
        dst = edge.get("dst_type", "")
        for match in lookup(src, etype, dst):
            if match.technique_id not in seen_ids:
                seen_ids.add(match.technique_id)
                results.append({
                    "technique_id": match.technique_id,
                    "technique_name": match.technique_name,
                    "tactic": match.tactic,
                    "rationale": match.rationale,
                })
    return results
