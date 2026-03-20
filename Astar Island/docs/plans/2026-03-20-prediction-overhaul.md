# Prediction System Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix critical bugs and add adaptive prediction to raise our Astar Island competition score from ~50 avg to 75+ avg.

**Architecture:** Fix the probability floor bug, add observation-driven adaptive priors that detect whether settlements thrive or collapse, lower prior strength so observations actually override priors, retrain the ridge model on all available data, and build a lightweight Norse civilization simulator for predicting unobserved cells.

**Tech Stack:** Python 3.12, numpy, existing `astar_island` package in `src/astar_island/`

**Test runner:** `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

**Important context:** The existing tests are in `tests/test_storage.py` and `tests/test_dataset.py`. Run all tests with `PYTHONPATH=src` because the package isn't installed editable.

---

### Task 1: Fix normalize_prediction probability floor bug

The floor is applied BEFORE normalization, so when a class has a large value (e.g. 100), the floor of 0.01 becomes 0.0001 after normalization. This caused an entire round to score 0.0001. The fix: apply floor, normalize, then repeat until stable.

**Files:**
- Modify: `src/astar_island/prediction.py:224-226`
- Create: `tests/test_prediction.py`

**Step 1: Write the failing test**

Create `tests/test_prediction.py`:

```python
from __future__ import annotations

import unittest

import numpy as np

from astar_island.prediction import (
    CLASS_COUNT,
    MIN_PROBABILITY,
    normalize_prediction,
    build_baseline_prediction,
    SeedPredictor,
)


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


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/test_prediction.py -v`

Expected: `test_floor_enforced_after_normalization` FAILS because current normalize_prediction produces min ~0.0001 for extreme inputs.

**Step 3: Fix normalize_prediction**

In `src/astar_island/prediction.py`, replace lines 224-226:

```python
def normalize_prediction(prediction: np.ndarray, *, floor: float = MIN_PROBABILITY) -> np.ndarray:
    clipped = np.maximum(prediction, floor)
    normalized = clipped / clipped.sum(axis=-1, keepdims=True)
    # Iterate: after normalization, some values may drop below floor.
    # Two passes is sufficient for convergence with 6 classes.
    for _ in range(2):
        below = normalized < floor
        if not below.any():
            break
        normalized = np.maximum(normalized, floor)
        normalized = normalized / normalized.sum(axis=-1, keepdims=True)
    return normalized
```

**Step 4: Run test to verify it passes**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/test_prediction.py -v`

Expected: ALL PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

Expected: ALL PASS

**Step 6: Commit**

```bash
git add tests/test_prediction.py src/astar_island/prediction.py
git commit -m "fix: enforce probability floor after normalization

The floor was applied before normalization, so extreme model outputs
(e.g. [100, 0, 0, 0, 0, 0]) would produce effective floors of 0.0001
instead of 0.01. This caused an entire round to score 0.0001.

Now iterates clip+normalize until the floor holds post-normalization."
```

---

### Task 2: Lower default prior strength from 20.0 to 4.0

With prior_strength=20, observations contribute only ~17% vs prior's 83% (at 4 observations per cell). Lowering to 4.0 makes observations dominant after just 4 samples.

**Files:**
- Modify: `src/astar_island/prediction.py:10` (change `DEFAULT_PRIOR_STRENGTH = 20.0` to `4.0`)
- Add tests to: `tests/test_prediction.py`

**Step 1: Write the failing test**

Add to `tests/test_prediction.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/test_prediction.py::SeedPredictorTests -v`

Expected: FAIL because prior_strength=20.0 means 4 observations barely shift the prior.

**Step 3: Change the default**

In `src/astar_island/prediction.py` line 10, change:
```python
DEFAULT_PRIOR_STRENGTH = 4.0
```

**Step 4: Run all tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/astar_island/prediction.py tests/test_prediction.py
git commit -m "fix: lower default prior strength from 20 to 4

With prior_strength=20, observations contributed only ~17% at 4
observations per cell. At 4.0, observations are equally weighted
with the prior after 4 samples, allowing real data to override
wrong heuristic priors."
```

---

### Task 3: Build adaptive prior system (detect collapse vs thrive)

This is the highest-impact feature. Observations reveal immediately whether settlements are thriving (371 alive in 5 observations) or collapsing (14 alive). We need to detect this and adjust priors for ALL cells, including unobserved ones.

**Files:**
- Create: `src/astar_island/dynamics.py`
- Create: `tests/test_dynamics.py`
- Modify: `src/astar_island/prediction.py:39-69` (add `settlement_survival` param to baseline builders)
- Modify: `src/astar_island/prediction.py:229-269` (`SeedPredictor` gets adaptive prior)

**Step 1: Create dynamics.py with observation signal extraction**

Create `src/astar_island/dynamics.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RoundDynamics:
    """Inferred round-level dynamics from observation signals."""
    settlement_alive_rate: float  # 0.0 = total collapse, 1.0 = all thrive
    settlement_density: float     # settlements per observed cell
    ruin_density: float           # ruins per observed cell
    port_rate: float              # fraction of settlements that are ports
    mean_population: float        # average settlement population
    mean_food: float              # average settlement food level
    observed_cells: int           # how many grid cells we've seen
    observed_queries: int         # how many queries contributed

    @property
    def is_collapse(self) -> bool:
        """Round where most settlements die."""
        return self.settlement_alive_rate < 0.15 and self.observed_queries >= 2

    @property
    def is_decline(self) -> bool:
        """Round with significant settlement loss."""
        return self.settlement_alive_rate < 0.4 and self.observed_queries >= 2

    @property
    def is_thriving(self) -> bool:
        """Round where settlements mostly survive."""
        return self.settlement_alive_rate > 0.7 and self.observed_queries >= 2

    @property
    def survival_factor(self) -> float:
        """Scaling factor for settlement survival predictions. 0.0-1.0."""
        if self.observed_queries < 1:
            return 0.5  # no data, assume moderate
        return min(1.0, max(0.0, self.settlement_alive_rate))


def extract_dynamics(observation_records: list[dict[str, Any]]) -> RoundDynamics:
    """Extract round-level dynamics from observation records (any seed)."""
    total_settlements_seen = 0
    alive_settlements = 0
    port_settlements = 0
    population_sum = 0.0
    food_sum = 0.0
    grid_cells_seen = 0
    settlement_cells = 0
    ruin_cells = 0
    queries = 0

    for record in observation_records:
        response = record.get("response", {})
        grid = response.get("grid", [])
        settlements = response.get("settlements", [])
        queries += 1

        for row in grid:
            for cell in row:
                grid_cells_seen += 1
                if cell == 1 or cell == 2:
                    settlement_cells += 1
                elif cell == 3:
                    ruin_cells += 1

        for s in settlements:
            total_settlements_seen += 1
            if s.get("alive", False):
                alive_settlements += 1
            if s.get("has_port", False):
                port_settlements += 1
            population_sum += float(s.get("population", 0.0))
            food_sum += float(s.get("food", 0.0))

    alive_rate = alive_settlements / max(1, total_settlements_seen)
    port_rate = port_settlements / max(1, alive_settlements)
    settlement_density = settlement_cells / max(1, grid_cells_seen)
    ruin_density = ruin_cells / max(1, grid_cells_seen)
    mean_pop = population_sum / max(1, total_settlements_seen)
    mean_food = food_sum / max(1, total_settlements_seen)

    return RoundDynamics(
        settlement_alive_rate=alive_rate,
        settlement_density=settlement_density,
        ruin_density=ruin_density,
        port_rate=port_rate,
        mean_population=mean_pop,
        mean_food=mean_food,
        observed_cells=grid_cells_seen,
        observed_queries=queries,
    )


def adjusted_class_priors(
    terrain_code: int,
    dynamics: RoundDynamics,
    *,
    is_coastal: bool = False,
) -> np.ndarray:
    """Return a 6-element prior adjusted by observed round dynamics.

    This replaces the fixed priors in build_baseline_prediction for
    dynamic terrain types (settlements, ports, ruins, plains, forests).
    Static terrain (ocean, mountain) is unchanged.
    """
    sf = dynamics.survival_factor

    if terrain_code == 5:  # Mountain - never changes
        return np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
    if terrain_code == 10:  # Ocean - never changes
        return np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])

    if terrain_code == 1:  # Settlement
        if dynamics.is_collapse:
            # Nearly all settlements die
            return np.array([0.55, 0.03, 0.01, 0.06, 0.33, 0.02])
        if dynamics.is_decline:
            return np.array([0.30, 0.15, 0.03, 0.20, 0.28, 0.04])
        # Thriving: settlements mostly survive
        survive = 0.10 + 0.45 * sf
        ruin = 0.05 + 0.20 * (1.0 - sf)
        empty = 0.05 + 0.25 * (1.0 - sf)
        forest = 0.05 + 0.10 * (1.0 - sf)
        return np.array([empty, survive, 0.05, ruin, forest, 0.02])

    if terrain_code == 2:  # Port
        if dynamics.is_collapse:
            return np.array([0.50, 0.02, 0.03, 0.08, 0.35, 0.02])
        if dynamics.is_decline:
            return np.array([0.25, 0.08, 0.15, 0.22, 0.26, 0.04])
        survive_port = 0.10 + 0.40 * sf
        return np.array([0.05, 0.08, survive_port, 0.15, 0.10, 0.02])

    if terrain_code == 3:  # Ruin
        if dynamics.is_collapse:
            return np.array([0.50, 0.01, 0.01, 0.05, 0.41, 0.02])
        # Ruins can be rebuilt if settlements thrive
        rebuild = 0.02 + 0.15 * sf
        return np.array([0.15, rebuild, 0.02, 0.40, 0.30, 0.02])

    if terrain_code == 4:  # Forest
        if dynamics.is_collapse:
            # Forests reclaim more land during collapse
            return np.array([0.04, 0.01, 0.01, 0.01, 0.92, 0.01])
        # Near settlements, forests may be cleared
        return np.array([0.05, 0.02 * sf, 0.01, 0.02, 0.88, 0.02])

    if terrain_code == 11:  # Plains
        if dynamics.is_collapse:
            # No settlement expansion during collapse
            return np.array([0.75, 0.02, 0.01, 0.02, 0.18, 0.02])
        if dynamics.is_decline:
            return np.array([0.60, 0.06, 0.02, 0.04, 0.24, 0.04])
        # Thriving: plains get settled
        settle = 0.05 + 0.15 * sf
        port = 0.02 + 0.05 * sf if is_coastal else 0.01
        return np.array([0.55 - settle, settle, port, 0.04, 0.15, 0.03])

    # Empty/other
    if dynamics.is_collapse:
        return np.array([0.80, 0.01, 0.01, 0.01, 0.15, 0.02])
    return np.array([0.70, 0.05, 0.03, 0.03, 0.14, 0.05])
```

**Step 2: Create tests/test_dynamics.py**

```python
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
        # In collapse, class 0 (empty) should dominate for settlements
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
```

**Step 3: Run dynamics tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/test_dynamics.py -v`

Expected: ALL PASS

**Step 4: Wire adaptive priors into SeedPredictor**

Modify `src/astar_island/prediction.py`:

Add import at top (after existing imports):
```python
from .dynamics import RoundDynamics, adjusted_class_priors, extract_dynamics
```

Add new method `build_adaptive_prediction` after `build_heuristic_prediction` (after line 221):

```python
def build_adaptive_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None = None,
    *,
    dynamics: RoundDynamics,
    floor: float = MIN_PROBABILITY,
) -> np.ndarray:
    """Build prediction using dynamics-adjusted priors instead of fixed ones."""
    resolved_grid = overlay_initial_settlements(initial_grid, settlements)
    height = len(resolved_grid)
    width = len(resolved_grid[0]) if height else 0
    coastal = _coastal_mask(resolved_grid)
    prediction = np.full((height, width, CLASS_COUNT), floor, dtype=np.float64)

    for y, row in enumerate(resolved_grid):
        for x, cell in enumerate(row):
            prediction[y, x] = adjusted_class_priors(
                cell, dynamics, is_coastal=bool(coastal[y, x] > 0)
            )

    return normalize_prediction(prediction, floor=floor)
```

Modify `SeedPredictor.from_initial_state` to accept optional dynamics (add after line 241):

Change the classmethod to:

```python
    @classmethod
    def from_initial_state(
        cls,
        initial_state: dict[str, Any],
        *,
        floor: float = MIN_PROBABILITY,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        heuristic: bool = True,
        dynamics: RoundDynamics | None = None,
    ) -> "SeedPredictor":
        if dynamics is not None:
            baseline = build_adaptive_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                dynamics=dynamics,
                floor=floor,
            )
        elif heuristic:
            baseline = build_heuristic_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                floor=floor,
            )
        else:
            baseline = build_baseline_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                floor=floor,
            )
        return cls(counts=baseline * prior_strength, floor=floor)
```

**Step 5: Add test for adaptive SeedPredictor**

Add to `tests/test_prediction.py`:

```python
from astar_island.dynamics import RoundDynamics


class AdaptivePredictorTests(unittest.TestCase):
    def test_collapse_dynamics_reduce_settlement_prediction(self):
        """With collapse dynamics, settlement cells should predict mostly empty/forest."""
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
```

**Step 6: Run all tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

Expected: ALL PASS

**Step 7: Commit**

```bash
git add src/astar_island/dynamics.py tests/test_dynamics.py tests/test_prediction.py src/astar_island/prediction.py
git commit -m "feat: add adaptive priors from observed round dynamics

Detects settlement survival rate from observations and adjusts priors
for all terrain types. In collapse rounds (0% survival), settlements
predict empty/forest instead of surviving. In thriving rounds (90%
survival), settlements predict high survival probability.

This addresses the biggest scoring gap: our fixed priors assumed ~40%
settlement survival, but rounds vary from 0% to 90%."
```

---

### Task 4: Wire adaptive dynamics into CLI run-baseline command

The CLI needs to use dynamics-adjusted priors. Strategy: do a first pass of observations, extract dynamics, then rebuild predictors with adaptive priors.

**Files:**
- Modify: `src/astar_island/cli.py:306-396` (`handle_run_baseline`)

**Step 1: Modify handle_run_baseline to use two-phase observation**

In `src/astar_island/cli.py`, replace the `handle_run_baseline` function:

```python
def handle_run_baseline(args: argparse.Namespace) -> None:
    from .dynamics import extract_dynamics

    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id, store=store, prefer_cache=True)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    budget = maybe_get_budget_for_round(client, round_id, store=store)
    if budget is not None:
        store.save_budget(round_id, budget)

    existing_observations = store.load_observations(round_id)
    existing_observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in existing_observations:
        request_payload = record.get("request", {})
        response_payload = record.get("response", {})
        seed_index = int(request_payload["seed_index"])
        existing_observations_by_seed.setdefault(seed_index, []).append(record)

    # Phase 1: Run observations
    if args.max_observations_per_seed > 0:
        if budget is None:
            raise SystemExit("Budget unavailable for this round; cannot run live observations.")
        schedule = round_robin_schedule(
            round_details,
            max_observations_per_seed=args.max_observations_per_seed,
            existing_observations_by_seed=existing_observations_by_seed,
            viewport_size=args.viewport_size,
        )
        remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        schedule = schedule[: max(0, remaining_budget)]

        for item in schedule:
            request_payload = {
                "round_id": round_id,
                "seed_index": item["seed_index"],
                "viewport_x": item["x"],
                "viewport_y": item["y"],
                "viewport_w": item["w"],
                "viewport_h": item["h"],
            }
            response_payload = client.simulate(
                round_id=round_id,
                seed_index=item["seed_index"],
                viewport_x=item["x"],
                viewport_y=item["y"],
                viewport_w=item["w"],
                viewport_h=item["h"],
            )
            record = {"request": request_payload, "response": response_payload}
            existing_observations.append(record)
            existing_observations_by_seed.setdefault(item["seed_index"], []).append(record)
            store.save_observation(round_id, request_payload, response_payload)
            store.save_budget(
                round_id,
                {
                    "queries_used": response_payload.get("queries_used"),
                    "queries_max": response_payload.get("queries_max"),
                },
            )
            time.sleep(max(0.0, args.simulate_delay_seconds))

    # Phase 2: Extract dynamics from ALL observations and rebuild predictors
    dynamics = extract_dynamics(existing_observations)
    predictors = [
        SeedPredictor.from_initial_state(initial_state, dynamics=dynamics)
        for initial_state in round_details["initial_states"]
    ]
    for record in existing_observations:
        request_payload = record.get("request", {})
        response_payload = record.get("response", {})
        seed_index = int(request_payload["seed_index"])
        predictors[seed_index].observe(response_payload)

    submitted = 0
    saved_predictions = []
    for seed_index, predictor in enumerate(predictors):
        prediction = predictor.prediction()
        metadata: dict[str, Any] = {
            "submitted": not args.no_submit,
            "dynamics": {
                "settlement_alive_rate": dynamics.settlement_alive_rate,
                "is_collapse": dynamics.is_collapse,
                "is_thriving": dynamics.is_thriving,
                "observed_queries": dynamics.observed_queries,
            },
        }
        if args.no_submit:
            submission_result = None
        else:
            submission_result = client.submit(round_id, seed_index, prediction.tolist())
            submitted += 1
            metadata["submission_result"] = submission_result
            if seed_index + 1 < len(predictors):
                time.sleep(max(0.0, args.submit_delay_seconds))

        path = store.save_prediction(round_id, seed_index, prediction, metadata=metadata)
        saved_predictions.append(str(path))

    print(
        json.dumps(
            {
                "round_id": round_id,
                "submitted": submitted,
                "dynamics": {
                    "settlement_alive_rate": dynamics.settlement_alive_rate,
                    "is_collapse": dynamics.is_collapse,
                    "is_thriving": dynamics.is_thriving,
                },
                "saved_predictions": saved_predictions,
            },
            indent=2,
        )
    )
```

**Step 2: Run all tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

Expected: ALL PASS (CLI handler changes don't break unit tests)

**Step 3: Commit**

```bash
git add src/astar_island/cli.py
git commit -m "feat: wire adaptive dynamics into run-baseline CLI

Extracts round dynamics from all observations before building
predictions. Detects collapse/thrive and adjusts priors accordingly.
Logs dynamics metadata in submission records."
```

---

### Task 5: Build Norse civilization simulator

The biggest impact feature. Simulates the game rules from CHALLENGE.md to predict what happens to unobserved cells. Uses initial state + inferred dynamics to run Monte Carlo simulations.

**Files:**
- Create: `src/astar_island/simulator.py`
- Create: `tests/test_simulator.py`
- Modify: `src/astar_island/prediction.py` (add `build_simulated_prediction`)
- Modify: `src/astar_island/cli.py` (add `run-simulate` command)

**Step 1: Create the simulator**

Create `src/astar_island/simulator.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .prediction import CLASS_COUNT, normalize_prediction, overlay_initial_settlements

YEARS = 50
DEFAULT_MONTE_CARLO_RUNS = 30


@dataclass
class SimSettlement:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int

    def copy(self) -> "SimSettlement":
        return SimSettlement(
            x=self.x, y=self.y, population=self.population,
            food=self.food, wealth=self.wealth, defense=self.defense,
            has_port=self.has_port, alive=self.alive, owner_id=self.owner_id,
        )


@dataclass
class SimParams:
    """Hidden parameters inferred from observations or set to defaults."""
    food_production: float = 0.3
    base_growth_rate: float = 0.15
    raid_probability: float = 0.3
    raid_strength: float = 0.4
    winter_severity: float = 0.25
    expansion_threshold: float = 2.5
    trade_range: int = 5
    trade_benefit: float = 0.1
    ruin_rebuild_chance: float = 0.05
    forest_reclaim_chance: float = 0.08
    starvation_threshold: float = 0.1

    @classmethod
    def from_dynamics(cls, dynamics: Any) -> "SimParams":
        """Infer simulation parameters from observed dynamics."""
        sf = dynamics.survival_factor
        if dynamics.is_collapse:
            return cls(
                food_production=0.10, base_growth_rate=0.05,
                raid_probability=0.6, raid_strength=0.7,
                winter_severity=0.5, expansion_threshold=5.0,
                ruin_rebuild_chance=0.01, forest_reclaim_chance=0.15,
            )
        if dynamics.is_decline:
            return cls(
                food_production=0.18, base_growth_rate=0.08,
                raid_probability=0.45, raid_strength=0.55,
                winter_severity=0.38, expansion_threshold=3.5,
                ruin_rebuild_chance=0.03, forest_reclaim_chance=0.12,
            )
        if dynamics.is_thriving:
            return cls(
                food_production=0.45, base_growth_rate=0.22,
                raid_probability=0.2, raid_strength=0.3,
                winter_severity=0.15, expansion_threshold=2.0,
                ruin_rebuild_chance=0.10, forest_reclaim_chance=0.04,
            )
        # Moderate
        return cls(
            food_production=0.25 + 0.15 * sf,
            base_growth_rate=0.10 + 0.10 * sf,
            raid_probability=0.25 + 0.15 * (1.0 - sf),
            raid_strength=0.35 + 0.15 * (1.0 - sf),
            winter_severity=0.20 + 0.15 * (1.0 - sf),
        )


def _adjacent_cells(x: int, y: int, width: int, height: int):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def _count_adjacent_food(grid: np.ndarray, x: int, y: int) -> int:
    h, w = grid.shape
    count = 0
    for nx, ny in _adjacent_cells(x, y, w, h):
        if grid[ny, nx] == 4:  # Forest provides food
            count += 1
    return count


def simulate_once(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None,
    params: SimParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run one Monte Carlo simulation, return final grid as class indices."""
    resolved = overlay_initial_settlements(initial_grid, settlements)
    height = len(resolved)
    width = len(resolved[0]) if height else 0
    grid = np.array(resolved, dtype=np.int16)

    # Initialize settlement objects
    sim_settlements: list[SimSettlement] = []
    next_owner = 0
    if settlements:
        for s in settlements:
            if not s.get("alive", True):
                continue
            sim_settlements.append(SimSettlement(
                x=int(s["x"]), y=int(s["y"]),
                population=float(s.get("population", 1.5)),
                food=float(s.get("food", 0.5)),
                wealth=float(s.get("wealth", 0.3)),
                defense=float(s.get("defense", 0.3)),
                has_port=bool(s.get("has_port", False)),
                alive=True,
                owner_id=int(s.get("owner_id", next_owner)),
            ))
            next_owner = max(next_owner, int(s.get("owner_id", next_owner)) + 1)

    for _year in range(YEARS):
        alive = [s for s in sim_settlements if s.alive]
        if not alive:
            break

        # 1. Growth: produce food from adjacent forests
        for s in alive:
            adj_food = _count_adjacent_food(grid, s.x, s.y)
            s.food += adj_food * params.food_production + params.food_production * 0.5
            if s.food > params.expansion_threshold and rng.random() < params.base_growth_rate:
                # Try to expand to adjacent empty land
                candidates = []
                for nx, ny in _adjacent_cells(s.x, s.y, width, height):
                    if grid[ny, nx] in (0, 11):  # empty or plains
                        candidates.append((nx, ny))
                if candidates:
                    ex, ey = candidates[rng.integers(len(candidates))]
                    is_coastal = any(
                        grid[ay, ax] == 10
                        for ax, ay in _adjacent_cells(ex, ey, width, height)
                    )
                    new_s = SimSettlement(
                        x=ex, y=ey, population=0.8, food=0.3,
                        wealth=0.1, defense=0.2,
                        has_port=is_coastal and rng.random() < 0.4,
                        alive=True, owner_id=s.owner_id,
                    )
                    sim_settlements.append(new_s)
                    grid[ey, ex] = 2 if new_s.has_port else 1
                    s.food -= 1.0
                    s.population -= 0.3

        # 2. Conflict: raids between settlements of different factions
        alive = [s for s in sim_settlements if s.alive]
        for attacker in alive:
            if rng.random() > params.raid_probability:
                continue
            targets = [
                t for t in alive
                if t.owner_id != attacker.owner_id
                and abs(t.x - attacker.x) + abs(t.y - attacker.y) <= 6
                and t.alive
            ]
            if not targets:
                continue
            target = targets[rng.integers(len(targets))]
            attack_power = attacker.population * params.raid_strength
            if attack_power > target.defense * target.population:
                target.owner_id = attacker.owner_id
                target.population *= 0.6
                target.food *= 0.5
                attacker.wealth += target.wealth * 0.3
            else:
                attacker.population *= 0.85

        # 3. Trade between allied ports
        ports = [s for s in sim_settlements if s.alive and s.has_port]
        for p in ports:
            partners = [
                t for t in ports
                if t is not p and t.owner_id == p.owner_id
                and abs(t.x - p.x) + abs(t.y - p.y) <= params.trade_range
            ]
            for partner in partners:
                p.food += params.trade_benefit
                p.wealth += params.trade_benefit * 0.5

        # 4. Winter: food consumption and potential collapse
        for s in sim_settlements:
            if not s.alive:
                continue
            s.food -= params.winter_severity * s.population
            s.population *= (1.0 - params.winter_severity * 0.1)
            if s.food < params.starvation_threshold and rng.random() < 0.3:
                s.alive = False
                grid[s.y, s.x] = 3  # becomes ruin
            elif s.population < 0.2:
                s.alive = False
                grid[s.y, s.x] = 3

        # 5. Environment: ruins reclaimed
        for y in range(height):
            for x in range(width):
                if grid[y, x] == 3:  # ruin
                    nearby_alive = any(
                        s.alive and abs(s.x - x) + abs(s.y - y) <= 2
                        for s in sim_settlements
                    )
                    if nearby_alive and rng.random() < params.ruin_rebuild_chance:
                        is_coastal = any(
                            grid[ay, ax] == 10
                            for ax, ay in _adjacent_cells(x, y, width, height)
                        )
                        grid[y, x] = 2 if is_coastal else 1
                    elif not nearby_alive and rng.random() < params.forest_reclaim_chance:
                        adj_forest = any(
                            grid[ay, ax] == 4
                            for ax, ay in _adjacent_cells(x, y, width, height)
                        )
                        if adj_forest:
                            grid[y, x] = 4
                        else:
                            grid[y, x] = 11  # back to plains

    # Convert to class indices
    from .prediction import terrain_code_to_class_index
    result = np.zeros((height, width), dtype=np.int16)
    for y in range(height):
        for x in range(width):
            result[y, x] = terrain_code_to_class_index(int(grid[y, x]))
    return result


def monte_carlo_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None,
    params: SimParams,
    *,
    runs: int = DEFAULT_MONTE_CARLO_RUNS,
    seed: int | None = None,
) -> np.ndarray:
    """Run multiple simulations and return probability distribution over classes."""
    rng = np.random.default_rng(seed)
    height = len(initial_grid)
    width = len(initial_grid[0]) if height else 0
    counts = np.zeros((height, width, CLASS_COUNT), dtype=np.float64)

    for _ in range(runs):
        result = simulate_once(initial_grid, settlements, params, rng)
        for y in range(height):
            for x in range(width):
                counts[y, x, result[y, x]] += 1.0

    return normalize_prediction(counts / max(1, runs))
```

**Step 2: Create tests/test_simulator.py**

```python
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
```

**Step 3: Run simulator tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/test_simulator.py -v`

Expected: ALL PASS

**Step 4: Add build_simulated_prediction to prediction.py**

Add after `build_adaptive_prediction` in `src/astar_island/prediction.py`:

```python
def build_simulated_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None = None,
    *,
    dynamics: RoundDynamics | None = None,
    runs: int = 30,
    floor: float = MIN_PROBABILITY,
) -> np.ndarray:
    """Build prediction by running Monte Carlo simulation."""
    from .simulator import SimParams, monte_carlo_prediction
    if dynamics is not None:
        params = SimParams.from_dynamics(dynamics)
    else:
        params = SimParams()
    return monte_carlo_prediction(initial_grid, settlements, params, runs=runs)
```

**Step 5: Wire simulator into SeedPredictor**

Update `SeedPredictor.from_initial_state` to accept `simulate` parameter:

```python
    @classmethod
    def from_initial_state(
        cls,
        initial_state: dict[str, Any],
        *,
        floor: float = MIN_PROBABILITY,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
        heuristic: bool = True,
        dynamics: RoundDynamics | None = None,
        simulate: bool = False,
        simulate_runs: int = 30,
    ) -> "SeedPredictor":
        if simulate:
            baseline = build_simulated_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                dynamics=dynamics,
                runs=simulate_runs,
                floor=floor,
            )
        elif dynamics is not None:
            baseline = build_adaptive_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                dynamics=dynamics,
                floor=floor,
            )
        elif heuristic:
            baseline = build_heuristic_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                floor=floor,
            )
        else:
            baseline = build_baseline_prediction(
                initial_state["grid"],
                initial_state.get("settlements"),
                floor=floor,
            )
        return cls(counts=baseline * prior_strength, floor=floor)
```

**Step 6: Add run-simulate CLI command**

Add to `src/astar_island/cli.py` in `build_parser` (after the run-baseline block, before analysis_parser):

```python
    sim_parser = subparsers.add_parser(
        "run-simulate",
        help="Build prediction using Monte Carlo simulation and submit.",
    )
    sim_parser.add_argument("--round-id", default=None)
    sim_parser.add_argument("--max-observations-per-seed", type=int, default=10)
    sim_parser.add_argument("--viewport-size", type=int, default=15)
    sim_parser.add_argument("--simulate-runs", type=int, default=30)
    sim_parser.add_argument("--simulate-delay-seconds", type=float, default=0.25)
    sim_parser.add_argument("--submit-delay-seconds", type=float, default=0.6)
    sim_parser.add_argument("--no-submit", action="store_true")
    sim_parser.set_defaults(handler=handle_run_simulate)
```

Add the handler function after `handle_run_baseline`:

```python
def handle_run_simulate(args: argparse.Namespace) -> None:
    from .dynamics import extract_dynamics

    client = build_client(args)
    store = build_store(args)
    round_details = resolve_round(client, args.round_id, store=store, prefer_cache=True)
    round_id = str(round_details["id"])
    store.save_round_details(round_details)

    budget = maybe_get_budget_for_round(client, round_id, store=store)
    if budget is not None:
        store.save_budget(round_id, budget)

    existing_observations = store.load_observations(round_id)
    existing_observations_by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in existing_observations:
        request_payload = record.get("request", {})
        seed_index = int(request_payload["seed_index"])
        existing_observations_by_seed.setdefault(seed_index, []).append(record)

    # Phase 1: Run observations
    if args.max_observations_per_seed > 0:
        if budget is None:
            raise SystemExit("Budget unavailable for this round; cannot run live observations.")
        schedule = round_robin_schedule(
            round_details,
            max_observations_per_seed=args.max_observations_per_seed,
            existing_observations_by_seed=existing_observations_by_seed,
            viewport_size=args.viewport_size,
        )
        remaining_budget = int(budget["queries_max"]) - int(budget["queries_used"])
        schedule = schedule[: max(0, remaining_budget)]

        for item in schedule:
            request_payload = {
                "round_id": round_id,
                "seed_index": item["seed_index"],
                "viewport_x": item["x"],
                "viewport_y": item["y"],
                "viewport_w": item["w"],
                "viewport_h": item["h"],
            }
            response_payload = client.simulate(
                round_id=round_id,
                seed_index=item["seed_index"],
                viewport_x=item["x"],
                viewport_y=item["y"],
                viewport_w=item["w"],
                viewport_h=item["h"],
            )
            record = {"request": request_payload, "response": response_payload}
            existing_observations.append(record)
            existing_observations_by_seed.setdefault(item["seed_index"], []).append(record)
            store.save_observation(round_id, request_payload, response_payload)
            store.save_budget(
                round_id,
                {
                    "queries_used": response_payload.get("queries_used"),
                    "queries_max": response_payload.get("queries_max"),
                },
            )
            time.sleep(max(0.0, args.simulate_delay_seconds))

    # Phase 2: Extract dynamics and build simulation-based predictions
    dynamics = extract_dynamics(existing_observations)
    predictors = [
        SeedPredictor.from_initial_state(
            initial_state, dynamics=dynamics,
            simulate=True, simulate_runs=args.simulate_runs,
        )
        for initial_state in round_details["initial_states"]
    ]
    for record in existing_observations:
        request_payload = record.get("request", {})
        response_payload = record.get("response", {})
        seed_index = int(request_payload["seed_index"])
        predictors[seed_index].observe(response_payload)

    submitted = 0
    saved_predictions = []
    for seed_index, predictor in enumerate(predictors):
        prediction = predictor.prediction()
        metadata: dict[str, Any] = {
            "submitted": not args.no_submit,
            "method": "simulate",
            "simulate_runs": args.simulate_runs,
            "dynamics": {
                "settlement_alive_rate": dynamics.settlement_alive_rate,
                "is_collapse": dynamics.is_collapse,
                "is_thriving": dynamics.is_thriving,
            },
        }
        if args.no_submit:
            submission_result = None
        else:
            submission_result = client.submit(round_id, seed_index, prediction.tolist())
            submitted += 1
            metadata["submission_result"] = submission_result
            if seed_index + 1 < len(predictors):
                time.sleep(max(0.0, args.submit_delay_seconds))

        path = store.save_prediction(round_id, seed_index, prediction, metadata=metadata)
        saved_predictions.append(str(path))

    print(
        json.dumps(
            {
                "round_id": round_id,
                "submitted": submitted,
                "method": "simulate",
                "simulate_runs": args.simulate_runs,
                "dynamics": {
                    "settlement_alive_rate": dynamics.settlement_alive_rate,
                    "is_collapse": dynamics.is_collapse,
                    "is_thriving": dynamics.is_thriving,
                },
                "saved_predictions": saved_predictions,
            },
            indent=2,
        )
    )
```

**Step 7: Run all tests**

Run: `cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -m pytest tests/ -v`

Expected: ALL PASS

**Step 8: Commit**

```bash
git add src/astar_island/simulator.py tests/test_simulator.py src/astar_island/prediction.py src/astar_island/cli.py
git commit -m "feat: add Norse civilization simulator for Monte Carlo predictions

Implements game rules from CHALLENGE.md: growth, conflict, trade,
winter, and environmental reclamation. SimParams are inferred from
observed dynamics (collapse/decline/thriving). Monte Carlo runs
produce probability distributions for all cells.

New CLI command: run-simulate"
```

---

### Task 6: Backtest against historical rounds

Validate improvements by scoring our new prediction methods against historical ground truth data. This doesn't need tests - it's a one-off analysis.

**Files:**
- None created; run as shell commands

**Step 1: Run backtesting script**

```bash
cd "D:\Koding\NM i AI\Astar Island" && PYTHONPATH=src python -c "
import json, numpy as np, math, glob
from astar_island.prediction import build_heuristic_prediction, build_adaptive_prediction, build_simulated_prediction, normalize_prediction
from astar_island.dynamics import extract_dynamics

def score(p, g):
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    p = normalize_prediction(p)
    g_safe = np.clip(g, 1e-12, 1.0)
    p_safe = np.clip(p, 1e-12, 1.0)
    entropy = -np.sum(g_safe * np.log(g_safe), axis=-1)
    kl = np.sum(g_safe * np.log(g_safe / p_safe), axis=-1)
    ws = entropy.sum()
    wkl = (entropy * kl).sum() / ws if ws > 0 else 0
    return max(0, min(100, 100 * math.exp(-3 * wkl)))

rounds_with_analysis = [
    '36e581f1-73f8-453f-ab98-cbe3052b701b',
    '71451d74-be9f-471f-aacd-a41f3b68a9cd',
    'ae78003a-4efe-425a-881a-d16a39bca0ad',
    'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
    'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
]

for round_id in rounds_with_analysis:
    details = json.load(open(f'data/rounds/{round_id}/round_details.json'))
    observations = [json.load(open(f)) for f in sorted(glob.glob(f'data/rounds/{round_id}/observations/q_*.json'))]
    dynamics = extract_dynamics(observations)

    heuristic_scores = []
    adaptive_scores = []
    simulated_scores = []
    api_scores = []

    for seed in range(5):
        a = json.load(open(f'data/rounds/{round_id}/analysis/seed_{seed}_analysis.json'))
        gt = np.array(a['ground_truth'], dtype=np.float64)
        api_scores.append(a['score'])

        initial_state = details['initial_states'][seed]
        h = build_heuristic_prediction(initial_state['grid'], initial_state.get('settlements'))
        heuristic_scores.append(score(h, gt))

        ad = build_adaptive_prediction(initial_state['grid'], initial_state.get('settlements'), dynamics=dynamics)
        adaptive_scores.append(score(ad, gt))

        sim = build_simulated_prediction(initial_state['grid'], initial_state.get('settlements'), dynamics=dynamics, runs=30)
        simulated_scores.append(score(sim, gt))

    print(f'Round {round_id[:8]} (collapse={dynamics.is_collapse} sf={dynamics.survival_factor:.2f}):')
    print(f'  API={np.mean(api_scores):5.1f}  heuristic={np.mean(heuristic_scores):5.1f}  adaptive={np.mean(adaptive_scores):5.1f}  simulated={np.mean(simulated_scores):5.1f}')
"
```

This will show whether our improvements help. Expected outcome:
- Adaptive should beat heuristic on collapse rounds significantly
- Simulated should beat both, especially on unobserved cells
- The gap should narrow from ~25 points to ~10-15

**Step 2: If backtesting looks good, commit a note**

No code changes needed - this validates the implementation.

---

### Summary of expected impact

| Change | Files | Expected Impact |
|--------|-------|-----------------|
| Task 1: Fix probability floor | `prediction.py` | Prevents 0-score catastrophes |
| Task 2: Lower prior strength | `prediction.py` | +3-5 pts (observations matter more) |
| Task 3: Adaptive priors | `dynamics.py`, `prediction.py` | +10-15 pts on collapse rounds |
| Task 4: Wire into CLI | `cli.py` | Enables using adaptive priors live |
| Task 5: Simulator | `simulator.py`, `prediction.py`, `cli.py` | +10-20 pts (predicts unobserved cells) |
| Task 6: Backtest | (none) | Validates improvements |
