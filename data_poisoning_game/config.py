from dataclasses import dataclass, field
from typing import List


@dataclass
class FLConfig:
    num_clients: int = 10
    num_rounds: int = 30
    local_epochs: int = 2
    local_batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    clients_per_round: int = 5
    lr_decay: float = 1.0


@dataclass
class ExperimentConfig:
    dataset: str = "cifar10"
    model: str = "cifar_cnn"
    dirichlet_alpha: float = 0.5
    adversarial_fraction: float = 0.2
    num_trials: int = 1
    device: str = "mps"
    seed: int = 42


@dataclass
class GameConfig:
    attacks: List[str] = field(default_factory=lambda: [
        "no_attack",
        "label_flip",
        "backdoor_pixel",
        "backdoor_edge_case",
        "model_scaling",
        "dba",
    ])
    defenses: List[str] = field(default_factory=lambda: [
        "fedavg",
        "krum",
        "multi_krum",
        "trimmed_mean",
        "coord_median",
        "norm_clip",
        "rfa",
    ])
    defense_cost_weight: float = 0.1
    fairness_weight: float = 0.05
    attack_cost_weight: float = 0.1
