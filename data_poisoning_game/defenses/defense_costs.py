DEFENSE_COSTS = {
    "fedavg": 0.0,
    "krum": 0.08,
    "multi_krum": 0.05,
    "trimmed_mean": 0.04,
    "coord_median": 0.06,
    "norm_clip": 0.03,
    "rfa": 0.07,
}


def get_defense_cost(defense_name: str) -> float:
    return DEFENSE_COSTS.get(defense_name, 0.0)
