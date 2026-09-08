import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


@dataclass
class PayoffMatrix:
    attacks: List[str]
    defenses: List[str]
    adversary_payoffs: np.ndarray
    server_payoffs: np.ndarray
    raw_accuracy: Optional[np.ndarray] = None
    raw_attack_success: Optional[np.ndarray] = None
    raw_fairness: Optional[np.ndarray] = None

    @classmethod
    def from_experiment_results(cls, results: Dict[Tuple[str, str], dict],
                                attacks: List[str], defenses: List[str],
                                defense_cost_weight: float = 0.1,
                                attack_cost_weight: float = 0.1,
                                fairness_weight: float = 0.05) -> "PayoffMatrix":
        m, n = len(attacks), len(defenses)
        adv_payoffs = np.zeros((m, n))
        srv_payoffs = np.zeros((m, n))
        accuracy_matrix = np.zeros((m, n))
        attack_success_matrix = np.zeros((m, n))
        fairness_matrix = np.zeros((m, n))

        from defenses.defense_costs import get_defense_cost
        from attacks import get_attack

        for i, atk in enumerate(attacks):
            attack_obj = get_attack(atk)
            for j, dfn in enumerate(defenses):
                key = (atk, dfn)
                if key not in results:
                    continue

                r = results[key]
                acc = r.get("accuracy", 0.0)
                asr = r.get("attack_success_rate", 0.0)
                worst_acc = r.get("worst_class_accuracy", acc)

                accuracy_matrix[i, j] = acc
                attack_success_matrix[i, j] = asr
                fairness_matrix[i, j] = acc - worst_acc

                defense_cost = get_defense_cost(dfn)
                attack_cost = attack_obj.cost

                adv_payoffs[i, j] = asr - attack_cost_weight * attack_cost
                if atk == "no_attack":
                    adv_payoffs[i, j] = 0.0

                srv_payoffs[i, j] = acc - defense_cost_weight * defense_cost - fairness_weight * (acc - worst_acc)

        return cls(
            attacks=attacks,
            defenses=defenses,
            adversary_payoffs=adv_payoffs,
            server_payoffs=srv_payoffs,
            raw_accuracy=accuracy_matrix,
            raw_attack_success=attack_success_matrix,
            raw_fairness=fairness_matrix,
        )

    @classmethod
    def from_arrays(cls, attacks: List[str], defenses: List[str],
                    adversary_payoffs: np.ndarray, server_payoffs: np.ndarray) -> "PayoffMatrix":
        return cls(attacks=attacks, defenses=defenses,
                   adversary_payoffs=adversary_payoffs, server_payoffs=server_payoffs)

    def to_dict(self) -> dict:
        return {
            "attacks": self.attacks,
            "defenses": self.defenses,
            "adversary_payoffs": self.adversary_payoffs.tolist(),
            "server_payoffs": self.server_payoffs.tolist(),
        }
