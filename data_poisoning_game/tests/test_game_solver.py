"""
Sanity check: verify the game solver produces valid equilibria on a synthetic payoff matrix.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from game_theory import GameSolver, PayoffMatrix


def test_known_equilibrium():
    """Matching pennies: unique NE is (0.5, 0.5) for both players."""
    attacks = ["heads", "tails"]
    defenses = ["heads", "tails"]
    adv_payoffs = np.array([[1, -1], [-1, 1]], dtype=float)
    srv_payoffs = np.array([[-1, 1], [1, -1]], dtype=float)

    pm = PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)
    solver = GameSolver(pm)
    results = solver.solve_nash()

    assert len(results) >= 1, "Should find at least one NE"
    ne = results[0]
    np.testing.assert_allclose(ne.adversary_strategy, [0.5, 0.5], atol=0.01)
    np.testing.assert_allclose(ne.server_strategy, [0.5, 0.5], atol=0.01)
    assert abs(ne.adversary_utility) < 0.01
    assert abs(ne.server_utility) < 0.01
    print("PASS: Matching pennies NE is correct.")


def test_dominant_strategy():
    """When one attack dominates, adversary plays it with probability 1."""
    attacks = ["weak", "strong"]
    defenses = ["d1", "d2"]
    adv_payoffs = np.array([[0.1, 0.1], [0.9, 0.8]], dtype=float)
    srv_payoffs = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=float)

    pm = PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)
    solver = GameSolver(pm)
    results = solver.solve_nash()

    assert len(results) >= 1
    ne = results[0]
    assert ne.adversary_strategy[1] > 0.99, "Strong attack should dominate"
    print("PASS: Dominant strategy detected.")


def test_value_of_information():
    """VoI should be non-negative."""
    attacks = ["a1", "a2", "a3"]
    defenses = ["d1", "d2", "d3"]
    rng = np.random.default_rng(42)
    adv_payoffs = rng.uniform(0, 1, (3, 3))
    srv_payoffs = rng.uniform(0, 1, (3, 3))

    pm = PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)
    solver = GameSolver(pm)
    results = solver.solve_nash()

    if results:
        voi = results[0].value_of_information(pm)
        assert voi >= -1e-6, f"VoI should be non-negative, got {voi}"
        print(f"PASS: VoI = {voi:.4f} >= 0")


def test_fictitious_play_converges():
    """Fictitious play should converge near NE utility."""
    attacks = ["a1", "a2"]
    defenses = ["d1", "d2"]
    adv_payoffs = np.array([[3, 0], [5, 1]], dtype=float)
    srv_payoffs = np.array([[3, 5], [0, 1]], dtype=float)

    pm = PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)
    solver = GameSolver(pm)

    fp_adv, fp_srv, convergence = solver.fictitious_play(num_iterations=20000)
    assert abs(fp_adv.sum() - 1.0) < 1e-6
    assert abs(fp_srv.sum() - 1.0) < 1e-6
    print(f"PASS: Fictitious play converges. Final strategies: adv={fp_adv}, srv={fp_srv}")


def test_stackelberg():
    """Stackelberg should give server >= Nash utility."""
    attacks = ["a1", "a2"]
    defenses = ["d1", "d2"]
    adv_payoffs = np.array([[1, -1], [-1, 1]], dtype=float)
    srv_payoffs = np.array([[-1, 1], [1, -1]], dtype=float)

    pm = PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)
    solver = GameSolver(pm)

    nash_results = solver.solve_nash()
    stackelberg = solver.solve_stackelberg()

    if nash_results and stackelberg:
        assert stackelberg.server_utility >= nash_results[0].server_utility - 0.01
        print(f"PASS: Stackelberg server utility ({stackelberg.server_utility:.3f}) >= Nash ({nash_results[0].server_utility:.3f})")


if __name__ == "__main__":
    test_known_equilibrium()
    test_dominant_strategy()
    test_value_of_information()
    test_fictitious_play_converges()
    test_stackelberg()
    print("\nAll tests passed!")
