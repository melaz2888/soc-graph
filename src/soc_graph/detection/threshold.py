from __future__ import annotations

from statistics import mean, pstdev


def sigma_threshold(scores: list[float], k: float = 3.0) -> float:
    if not scores:
        raise ValueError("scores must not be empty")
    if len(scores) == 1:
        return scores[0]
    return mean(scores) + k * pstdev(scores)


def flag_scores(score_map: dict[str, float], threshold: float) -> dict[str, float]:
    return {key: value for key, value in score_map.items() if value > threshold}

