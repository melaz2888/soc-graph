from soc_graph.detection.threshold import flag_scores, sigma_threshold


def test_sigma_threshold_and_flagging() -> None:
    scores = [0.1, 0.2, 0.25, 2.0]
    threshold = sigma_threshold(scores, k=1.0)

    flagged = flag_scores({"a": 0.2, "b": 2.0}, threshold)

    assert "b" in flagged
    assert "a" not in flagged

