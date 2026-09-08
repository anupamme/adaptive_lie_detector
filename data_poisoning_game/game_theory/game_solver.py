import numpy as np
import nashpy as nash
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .payoff_matrix import PayoffMatrix


@dataclass
class GameResult:
    adversary_strategy: np.ndarray
    server_strategy: np.ndarray
    adversary_utility: float
    server_utility: float
    equilibrium_type: str
    attacks: List[str]
    defenses: List[str]

    @property
    def adversary_support(self) -> List[Tuple[str, float]]:
        return [(self.attacks[i], p) for i, p in enumerate(self.adversary_strategy) if p > 1e-6]

    @property
    def server_support(self) -> List[Tuple[str, float]]:
        return [(self.defenses[j], p) for j, p in enumerate(self.server_strategy) if p > 1e-6]

    def value_of_information(self, payoff_matrix: PayoffMatrix) -> float:
        best_response_utility = 0.0
        for j in range(len(self.defenses)):
            if self.server_strategy[j] > 1e-6:
                br_utility = payoff_matrix.adversary_payoffs[:, j].max()
                best_response_utility += self.server_strategy[j] * br_utility
        return best_response_utility - self.adversary_utility


class GameSolver:
    def __init__(self, payoff_matrix: PayoffMatrix):
        self.payoff_matrix = payoff_matrix
        self.game = nash.Game(payoff_matrix.adversary_payoffs, payoff_matrix.server_payoffs)

    def solve_nash(self) -> List[GameResult]:
        results = []
        try:
            equilibria = list(self.game.support_enumeration())
        except Exception:
            equilibria = list(self.game.vertex_enumeration())

        for eq in equilibria:
            adv_strat, srv_strat = eq
            if not self._is_valid_strategy(adv_strat) or not self._is_valid_strategy(srv_strat):
                continue

            adv_util = float(adv_strat @ self.payoff_matrix.adversary_payoffs @ srv_strat)
            srv_util = float(adv_strat @ self.payoff_matrix.server_payoffs @ srv_strat)

            results.append(GameResult(
                adversary_strategy=adv_strat,
                server_strategy=srv_strat,
                adversary_utility=adv_util,
                server_utility=srv_util,
                equilibrium_type="nash",
                attacks=self.payoff_matrix.attacks,
                defenses=self.payoff_matrix.defenses,
            ))
        return results

    def solve_maxmin(self) -> GameResult:
        from scipy.optimize import linprog

        m = len(self.payoff_matrix.attacks)
        n = len(self.payoff_matrix.defenses)
        A = self.payoff_matrix.adversary_payoffs

        c = np.zeros(m + 1)
        c[-1] = -1.0

        A_ub = np.zeros((n, m + 1))
        for j in range(n):
            A_ub[j, :m] = -A[:, j]
            A_ub[j, -1] = 1.0
        b_ub = np.zeros(n)

        A_eq = np.zeros((1, m + 1))
        A_eq[0, :m] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * m + [(None, None)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        adv_strat = result.x[:m]
        adv_util = -result.fun

        srv_strat = self._best_response_server(adv_strat)
        srv_util = float(adv_strat @ self.payoff_matrix.server_payoffs @ srv_strat)

        return GameResult(
            adversary_strategy=adv_strat,
            server_strategy=srv_strat,
            adversary_utility=adv_util,
            server_utility=srv_util,
            equilibrium_type="maxmin",
            attacks=self.payoff_matrix.attacks,
            defenses=self.payoff_matrix.defenses,
        )

    def solve_stackelberg(self) -> GameResult:
        n = len(self.payoff_matrix.defenses)
        m = len(self.payoff_matrix.attacks)

        best_srv_util = -np.inf
        best_result = None

        for follower_action in range(m):
            from scipy.optimize import linprog

            c = -self.payoff_matrix.server_payoffs[follower_action, :]

            A_ub = []
            b_ub = []
            for i in range(m):
                if i == follower_action:
                    continue
                constraint = (self.payoff_matrix.adversary_payoffs[follower_action, :]
                              - self.payoff_matrix.adversary_payoffs[i, :])
                A_ub.append(-constraint)
                b_ub.append(0.0)

            A_eq = [np.ones(n)]
            b_eq = [1.0]
            bounds = [(0, 1)] * n

            if A_ub:
                result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                                 A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds)
            else:
                result = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds)

            if result.success:
                srv_util = -result.fun
                if srv_util > best_srv_util:
                    best_srv_util = srv_util
                    srv_strat = result.x
                    adv_strat = np.zeros(m)
                    adv_strat[follower_action] = 1.0
                    adv_util = float(adv_strat @ self.payoff_matrix.adversary_payoffs @ srv_strat)
                    best_result = GameResult(
                        adversary_strategy=adv_strat,
                        server_strategy=srv_strat,
                        adversary_utility=adv_util,
                        server_utility=best_srv_util,
                        equilibrium_type="stackelberg",
                        attacks=self.payoff_matrix.attacks,
                        defenses=self.payoff_matrix.defenses,
                    )

        return best_result

    def fictitious_play(self, num_iterations: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        m = len(self.payoff_matrix.attacks)
        n = len(self.payoff_matrix.defenses)

        adv_counts = np.zeros(m)
        srv_counts = np.zeros(n)
        adv_counts[0] = 1
        srv_counts[0] = 1

        convergence = []

        for t in range(1, num_iterations):
            srv_mixed = srv_counts / srv_counts.sum()
            adv_payoffs = self.payoff_matrix.adversary_payoffs @ srv_mixed
            adv_br = np.argmax(adv_payoffs)
            adv_counts[adv_br] += 1

            adv_mixed = adv_counts / adv_counts.sum()
            srv_payoffs = self.payoff_matrix.server_payoffs.T @ adv_mixed
            srv_br = np.argmax(srv_payoffs)
            srv_counts[srv_br] += 1

            if t % 100 == 0:
                adv_mixed = adv_counts / adv_counts.sum()
                srv_mixed = srv_counts / srv_counts.sum()
                util = float(adv_mixed @ self.payoff_matrix.adversary_payoffs @ srv_mixed)
                convergence.append(util)

        adv_strategy = adv_counts / adv_counts.sum()
        srv_strategy = srv_counts / srv_counts.sum()
        return adv_strategy, srv_strategy, convergence

    def _best_response_server(self, adv_strategy: np.ndarray) -> np.ndarray:
        expected_payoffs = self.payoff_matrix.server_payoffs.T @ adv_strategy
        srv_strat = np.zeros(len(self.payoff_matrix.defenses))
        srv_strat[np.argmax(expected_payoffs)] = 1.0
        return srv_strat

    @staticmethod
    def _is_valid_strategy(strategy: np.ndarray) -> bool:
        return np.all(strategy >= -1e-6) and abs(strategy.sum() - 1.0) < 1e-4
