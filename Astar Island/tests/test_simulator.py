from __future__ import annotations

import unittest

import numpy as np

from astar_island.simulator import (
    SimParams,
    simulate_once,
    monte_carlo_prediction,
)


class SimulateOnceTests(unittest.TestCase):
    def test_static_map_stays_static(self):
        """A map with only ocean and mountains shouldn't change."""
        grid = [[10, 10], [5, 5]]
        result = simulate_once(grid, [], SimParams(), np.random.default_rng(42))
        # Ocean=class 0, Mountain=class 5
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(result[1, 0], 5)
        self.assertEqual(result[1, 1], 5)

    def test_settlement_can_survive_or_die(self):
        """Settlement should either survive (class 1/2) or become ruin/empty (class 0/3/4)."""
        grid = [[11, 11, 11], [11, 1, 11], [11, 4, 11]]
        settlements = [{"x": 1, "y": 1, "has_port": False, "alive": True}]
        result = simulate_once(grid, settlements, SimParams(), np.random.default_rng(42))
        # Cell (1,1) should be some valid class
        self.assertIn(result[1, 1], [0, 1, 2, 3, 4])

    def test_collapse_params_kill_settlements(self):
        """With extreme collapse params, settlements should die."""
        grid = [[11, 11, 11], [11, 1, 11], [11, 11, 11]]
        settlements = [{"x": 1, "y": 1, "has_port": False, "alive": True}]
        collapse = SimParams(
            food_production=0.01, raid_probability=0.9,
            raid_strength=0.9, winter_severity=0.8,
        )
        deaths = 0
        rng = np.random.default_rng(42)
        for _ in range(20):
            result = simulate_once(grid, settlements, collapse, rng)
            if result[1, 1] != 1:  # not settlement anymore
                deaths += 1
        self.assertGreater(deaths, 10, "Most runs should kill the settlement with collapse params")


class MonteCarloTests(unittest.TestCase):
    def test_output_shape_and_normalization(self):
        grid = [[11, 11], [11, 11]]
        pred = monte_carlo_prediction(grid, None, SimParams(), runs=5, seed=42)
        self.assertEqual(pred.shape, (2, 2, 6))
        np.testing.assert_allclose(pred.sum(axis=-1), 1.0, atol=1e-6)
        self.assertTrue(np.all(pred >= 0.01 - 1e-9))

    def test_ocean_prediction_confident(self):
        grid = [[10, 10], [10, 10]]
        pred = monte_carlo_prediction(grid, None, SimParams(), runs=10, seed=42)
        # Ocean should be confidently class 0
        for y in range(2):
            for x in range(2):
                self.assertGreater(pred[y, x, 0], 0.8)


if __name__ == "__main__":
    unittest.main()
