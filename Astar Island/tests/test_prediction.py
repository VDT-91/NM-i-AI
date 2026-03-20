from __future__ import annotations

import unittest

import numpy as np

from astar_island.prediction import (
    CLASS_COUNT,
    MIN_PROBABILITY,
    SeedPredictor,
    build_baseline_prediction,
    build_heuristic_prediction,
    normalize_prediction,
    overlay_initial_settlements,
    terrain_code_to_class_index,
)


class PredictionTests(unittest.TestCase):
    def test_terrain_code_mapping(self) -> None:
        self.assertEqual(terrain_code_to_class_index(10), 0)
        self.assertEqual(terrain_code_to_class_index(11), 0)
        self.assertEqual(terrain_code_to_class_index(0), 0)
        self.assertEqual(terrain_code_to_class_index(5), 5)

    def test_overlay_initial_settlements(self) -> None:
        grid = [[11, 11], [11, 11]]
        settlements = [
            {"x": 0, "y": 1, "has_port": False, "alive": True},
            {"x": 1, "y": 0, "has_port": True, "alive": True},
        ]
        resolved = overlay_initial_settlements(grid, settlements)
        self.assertEqual(resolved[1][0], 1)
        self.assertEqual(resolved[0][1], 2)
        self.assertEqual(grid[1][0], 11)

    def test_baseline_prediction_is_normalized(self) -> None:
        prediction = build_baseline_prediction(
            [[10, 5], [4, 11]],
            settlements=[{"x": 1, "y": 1, "has_port": False, "alive": True}],
        )
        self.assertEqual(prediction.shape, (2, 2, 6))
        np.testing.assert_allclose(prediction.sum(axis=-1), np.ones((2, 2)))

    def test_heuristic_prediction_boosts_dynamic_cells_near_settlements(self) -> None:
        prediction = build_heuristic_prediction(
            [[10, 11, 11], [4, 11, 11]],
            settlements=[{"x": 2, "y": 1, "has_port": True, "alive": True}],
        )
        self.assertEqual(prediction.shape, (2, 3, 6))
        np.testing.assert_allclose(prediction.sum(axis=-1), np.ones((2, 3)))
        self.assertGreater(prediction[0, 1, 1], 0.10)
        self.assertGreater(prediction[1, 1, 2], 0.05)
        self.assertEqual(int(prediction[0, 0].argmax()), 0)

    def test_observation_updates_cell_probability(self) -> None:
        predictor = SeedPredictor.from_initial_state(
            {
                "grid": [[11, 11], [11, 11]],
                "settlements": [],
            },
            prior_strength=1.0,
        )
        predictor.observe(
            {
                "viewport": {"x": 0, "y": 0, "w": 1, "h": 1},
                "grid": [[5]],
            }
        )
        prediction = predictor.prediction()
        self.assertGreater(prediction[0, 0, 5], prediction[0, 0, 0])


class NormalizePredictionTests(unittest.TestCase):
    def test_floor_enforced_after_normalization(self):
        """When one class dominates, floor must still hold after normalization."""
        extreme = np.array([[[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        result = normalize_prediction(extreme)
        self.assertTrue(
            np.all(result >= MIN_PROBABILITY - 1e-9),
            f"Min value {result.min()} is below floor {MIN_PROBABILITY}",
        )
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_floor_with_negative_inputs(self):
        """Negative values should be clipped to floor."""
        negative = np.array([[[-5.0, 0.0, 0.0, 0.0, 0.0, 10.0]]])
        result = normalize_prediction(negative)
        self.assertTrue(np.all(result >= MIN_PROBABILITY - 1e-9))
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_already_valid_prediction_unchanged(self):
        """A valid prediction that already meets floor should pass through."""
        valid = np.array([[[0.80, 0.04, 0.04, 0.04, 0.04, 0.04]]])
        result = normalize_prediction(valid)
        np.testing.assert_allclose(result, valid, atol=1e-6)

    def test_batch_floor_enforcement(self):
        """Floor must hold for every cell in a batch."""
        batch = np.zeros((5, 5, CLASS_COUNT))
        batch[:, :, 0] = 1000.0  # extreme dominance
        result = normalize_prediction(batch)
        self.assertTrue(np.all(result >= MIN_PROBABILITY - 1e-9))
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-6)

    def test_uniform_stays_uniform(self):
        """Uniform distribution should remain uniform."""
        uniform = np.full((2, 2, CLASS_COUNT), 1.0 / CLASS_COUNT)
        result = normalize_prediction(uniform)
        np.testing.assert_allclose(result, uniform, atol=1e-6)


class SeedPredictorTests(unittest.TestCase):
    def test_observations_override_prior(self):
        """After 4 observations of a cell as class 0, prediction should favor class 0."""
        initial_state = {
            "grid": [[11, 11], [11, 11]],
            "settlements": [{"x": 0, "y": 0, "has_port": False, "alive": True}],
        }
        predictor = SeedPredictor.from_initial_state(initial_state)

        # Observe cell (0,0) as class 0 (empty) four times
        for _ in range(4):
            predictor.observe({
                "viewport": {"x": 0, "y": 0, "w": 1, "h": 1},
                "grid": [[0]],
            })

        pred = predictor.prediction()
        # After 4 observations of empty, class 0 should dominate
        # even though initial state was settlement (class 1)
        self.assertGreater(
            pred[0, 0, 0], 0.35,
            f"Class 0 should dominate after 4 observations, got {pred[0, 0, 0]:.3f}",
        )


class AdaptivePredictorTests(unittest.TestCase):
    def test_collapse_dynamics_reduce_settlement_prediction(self):
        """With collapse dynamics, settlement cells should predict mostly empty/forest."""
        from astar_island.dynamics import RoundDynamics

        initial_state = {
            "grid": [[1, 11], [4, 11]],
            "settlements": [{"x": 0, "y": 0, "has_port": False, "alive": True}],
        }
        collapse = RoundDynamics(
            settlement_alive_rate=0.05, settlement_density=0.01,
            ruin_density=0.02, port_rate=0.0, mean_population=0.1,
            mean_food=0.0, observed_cells=500, observed_queries=10,
        )
        predictor = SeedPredictor.from_initial_state(initial_state, dynamics=collapse)
        pred = predictor.prediction()
        # Settlement cell should predict mostly empty in collapse
        self.assertGreater(pred[0, 0, 0], 0.35, "Class 0 should dominate for settlements in collapse")
        self.assertLess(pred[0, 0, 1], 0.15, "Class 1 (settlement) should be low in collapse")


if __name__ == "__main__":
    unittest.main()
