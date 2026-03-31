from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TorchBackendStatus:
    available: bool
    detail: str


def check_torch_backend() -> TorchBackendStatus:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return TorchBackendStatus(available=False, detail=str(exc))
    return TorchBackendStatus(available=True, detail="torch and torch_geometric are available")

