from __future__ import annotations

import unittest

import numpy as np

from astar_island.dynamics import (
    RoundDynamics,
    extract_dynamics,
    adjusted_class_priors,
)


def _make_observation(grid, settlements=None, seed_index=0):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return {
        "request": {"seed_index": seed_index, "viewport_x": 0, "viewport_y": 0, "viewport_w": w, "viewport_h": h},
        "response": {
            "grid": grid,
            "settlements": settlements or [],
            "viewport": {"x": 0, "y": 0, "w": w, "h": h},
        },
    }


class ExtractDynamicsTests(unittest.TestCase):
    def test_thriving_round(self):
        obs = [
            _make_observation(
                [[1, 1], [11, 4]],
                [
                    {"x": 0, "y": 0, "alive": True, "has_port": False, "population": 3.0, "food": 0.5},
                    {"x": 1, "y": 0, "alive": True, "has_port": False, "population": 2.5, "food": 0.4},
                ],
            )
            for _ in range(3)
        ]
        d = extract_dynamics(obs)
        self.assertGreater(d.settlement_alive_rate, 0.9)
        self.assertTrue(d.is_thriving)
        self.assertFalse(d.is_collapse)

    def test_collapse_round(self):
        obs = [
            _make_observation(
                [[0, 0], [3, 4]],
                [
                    {"x": 0, "y": 1, "alive": False, "has_port": False, "population": 0.0, "food": 0.0},
                ],
            )
            for _ in range(3)
        ]
        d = extract_dynamics(obs)
        self.assertLess(d.settlement_alive_rate, 0.1)
        self.assertTrue(d.is_collapse)

    def test_no_observations(self):
        d = extract_dynamics([])
        self.assertEqual(d.observed_queries, 0)
        self.assertAlmostEqual(d.survival_factor, 0.5)

    def test_settlement_priors_collapse(self):
        collapse = RoundDynamics(
            settlement_alive_rate=0.05, settlement_density=0.01,
            ruin_density=0.02, port_rate=0.0, mean_population=0.1,
            mean_food=0.0, observed_cells=100, observed_queries=5,
        )
        priors = adjusted_class_priors(1, collapse)  # settlement
        self.assertGreater(priors[0], 0.4, "Empty should be most likely in collapse")
        self.assertLess(priors[1], 0.10, "Settlement survival should be low")
        np.testing.assert_allclose(priors.sum(), 1.0, atol=0.02)

    def test_settlement_priors_thriving(self):
        thriving = RoundDynamics(
            settlement_alive_rate=0.9, settlement_density=0.15,
            ruin_density=0.01, port_rate=0.2, mean_population=3.0,
            mean_food=0.5, observed_cells=500, observed_queries=10,
        )
        priors = adjusted_class_priors(1, thriving)
        self.assertGreater(priors[1], 0.35, "Settlement should likely survive when thriving")
        np.testing.assert_allclose(priors.sum(), 1.0, atol=0.02)

    def test_static_terrain_unchanged(self):
        collapse = RoundDynamics(
            settlement_alive_rate=0.0, settlement_density=0.0,
            ruin_density=0.0, port_rate=0.0, mean_population=0.0,
            mean_food=0.0, observed_cells=100, observed_queries=5,
        )
        mountain = adjusted_class_priors(5, collapse)
        self.assertGreater(mountain[5], 0.9)
        ocean = adjusted_class_priors(10, collapse)
        self.assertGreater(ocean[0], 0.9)


if __name__ == "__main__":
    unittest.main()
