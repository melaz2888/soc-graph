from soc_graph.model.runtime import check_torch_backend


def test_torch_backend_check_returns_status() -> None:
    status = check_torch_backend()
    assert isinstance(status.available, bool)
    assert isinstance(status.detail, str)

