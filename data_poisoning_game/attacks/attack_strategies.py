import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_fn, poison_fraction: float = 1.0, seed: int = 42):
        self.base_dataset = base_dataset
        self.poison_fn = poison_fn
        self.poison_fraction = poison_fraction
        rng = np.random.default_rng(seed)
        n = len(base_dataset)
        num_poison = int(n * poison_fraction)
        self.poison_indices = set(rng.choice(n, num_poison, replace=False).tolist())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        if idx in self.poison_indices:
            data, target = self.poison_fn(data, target)
        return data, target


def label_flip_fn(source_class: int = 0, target_class: int = 1):
    def fn(data, target):
        if target == source_class:
            return data, target_class
        return data, target
    return fn


def backdoor_pixel_fn(trigger_size: int = 4, target_class: int = 0):
    def fn(data, target):
        poisoned = data.clone()
        poisoned[:, -trigger_size:, -trigger_size:] = 1.0
        return poisoned, target_class
    return fn


def backdoor_edge_case_fn(target_class: int = 0):
    def fn(data, target):
        poisoned = data.clone()
        poisoned[:, 0:2, 0:2] = -1.0
        poisoned[:, 0:2, -2:] = -1.0
        return poisoned, target_class
    return fn


class AttackStrategy:
    def __init__(self, name: str):
        self.name = name

    def poison_dataset(self, dataset) -> Dataset:
        return dataset

    def manipulate_update(self, update: Dict[str, torch.Tensor],
                          global_model: nn.Module) -> Dict[str, torch.Tensor]:
        return update

    @property
    def cost(self) -> float:
        return 0.0


class NoAttack(AttackStrategy):
    def __init__(self):
        super().__init__("no_attack")

    @property
    def cost(self):
        return 0.0


class LabelFlipAttack(AttackStrategy):
    def __init__(self, source_class: int = 0, target_class: int = 1, poison_fraction: float = 1.0):
        super().__init__("label_flip")
        self.source_class = source_class
        self.target_class = target_class
        self.poison_fraction = poison_fraction

    def poison_dataset(self, dataset) -> Dataset:
        return PoisonedDataset(dataset, label_flip_fn(self.source_class, self.target_class), self.poison_fraction)

    @property
    def cost(self):
        return 0.1


class BackdoorPixelAttack(AttackStrategy):
    def __init__(self, trigger_size: int = 4, target_class: int = 0, poison_fraction: float = 0.5):
        super().__init__("backdoor_pixel")
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.poison_fraction = poison_fraction

    def poison_dataset(self, dataset) -> Dataset:
        return PoisonedDataset(dataset, backdoor_pixel_fn(self.trigger_size, self.target_class), self.poison_fraction)

    @property
    def cost(self):
        return 0.2


class BackdoorEdgeCaseAttack(AttackStrategy):
    def __init__(self, target_class: int = 0, poison_fraction: float = 0.3):
        super().__init__("backdoor_edge_case")
        self.target_class = target_class
        self.poison_fraction = poison_fraction

    def poison_dataset(self, dataset) -> Dataset:
        return PoisonedDataset(dataset, backdoor_edge_case_fn(self.target_class), self.poison_fraction)

    @property
    def cost(self):
        return 0.15


class ModelScalingAttack(AttackStrategy):
    def __init__(self, scale_factor: float = 10.0, target_class: int = 0,
                 trigger_size: int = 4, poison_fraction: float = 0.5):
        super().__init__("model_scaling")
        self.scale_factor = scale_factor
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.poison_fraction = poison_fraction

    def poison_dataset(self, dataset) -> Dataset:
        return PoisonedDataset(dataset, backdoor_pixel_fn(self.trigger_size, self.target_class), self.poison_fraction)

    def manipulate_update(self, update: Dict[str, torch.Tensor],
                          global_model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: param * self.scale_factor for name, param in update.items()}

    @property
    def cost(self):
        return 0.3


class DBAAttack(AttackStrategy):
    def __init__(self, num_triggers: int = 4, target_class: int = 0, poison_fraction: float = 0.5):
        super().__init__("dba")
        self.num_triggers = num_triggers
        self.target_class = target_class
        self.poison_fraction = poison_fraction
        self._trigger_idx = 0

    def get_trigger_fn(self, trigger_idx: int):
        positions = [(0, 0), (0, -3), (-3, 0), (-3, -3)]
        row, col = positions[trigger_idx % len(positions)]

        def fn(data, target):
            poisoned = data.clone()
            poisoned[:, row:row + 3 if row >= 0 else None, col:col + 3 if col >= 0 else None] = 1.0
            return poisoned, self.target_class
        return fn

    def poison_dataset(self, dataset) -> Dataset:
        fn = self.get_trigger_fn(self._trigger_idx)
        self._trigger_idx = (self._trigger_idx + 1) % self.num_triggers
        return PoisonedDataset(dataset, fn, self.poison_fraction)

    @property
    def cost(self):
        return 0.25


ATTACK_REGISTRY = {
    "no_attack": NoAttack,
    "label_flip": LabelFlipAttack,
    "backdoor_pixel": BackdoorPixelAttack,
    "backdoor_edge_case": BackdoorEdgeCaseAttack,
    "model_scaling": ModelScalingAttack,
    "dba": DBAAttack,
}


def get_attack(name: str, **kwargs) -> AttackStrategy:
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[name](**kwargs)
