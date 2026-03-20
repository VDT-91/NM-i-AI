from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dynamics import RoundDynamics, adjusted_class_priors

CLASS_COUNT = 6
MIN_PROBABILITY = 0.01
DEFAULT_PRIOR_STRENGTH = 4.0
MAX_INFLUENCE_DISTANCE = 8


def terrain_code_to_class_index(code: int) -> int:
    if code in {0, 10, 11}:
        return 0
    if code in {1, 2, 3, 4, 5}:
        return code
    raise ValueError(f"Unsupported terrain code: {code}")


def overlay_initial_settlements(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None,
) -> list[list[int]]:
    grid = [row[:] for row in initial_grid]
    if not settlements:
        return grid

    for settlement in settlements:
        if not settlement.get("alive", True):
            continue
        x = int(settlement["x"])
        y = int(settlement["y"])
        grid[y][x] = 2 if settlement.get("has_port") else 1
    return grid


def build_baseline_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None = None,
    *,
    floor: float = MIN_PROBABILITY,
) -> np.ndarray:
    resolved_grid = overlay_initial_settlements(initial_grid, settlements)
    height = len(resolved_grid)
    width = len(resolved_grid[0]) if height else 0
    prediction = np.full((height, width, CLASS_COUNT), floor, dtype=np.float64)

    for y, row in enumerate(resolved_grid):
        for x, cell in enumerate(row):
            if cell == 5:
                prediction[y, x] = [floor, floor, floor, floor, floor, 0.95]
            elif cell == 10:
                prediction[y, x] = [0.95, floor, floor, floor, floor, floor]
            elif cell == 4:
                prediction[y, x] = [0.05, 0.02, floor, 0.02, 0.88, 0.02]
            elif cell == 1:
                prediction[y, x] = [0.05, 0.40, 0.10, 0.25, 0.10, 0.10]
            elif cell == 2:
                prediction[y, x] = [0.05, 0.10, 0.40, 0.25, 0.10, 0.10]
            elif cell == 3:
                prediction[y, x] = [0.10, 0.05, 0.05, 0.55, 0.20, 0.05]
            elif cell == 11:
                prediction[y, x] = [0.60, 0.10, 0.05, 0.05, 0.15, 0.05]
            else:
                prediction[y, x] = [0.70, 0.05, 0.05, 0.05, 0.10, 0.05]

    return normalize_prediction(prediction, floor=floor)


def _neighbor_coordinates(x: int, y: int, width: int, height: int):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height:
            yield nx, ny


def _coastal_mask(resolved_grid: list[list[int]]) -> np.ndarray:
    height = len(resolved_grid)
    width = len(resolved_grid[0]) if height else 0
    coastal = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            if int(resolved_grid[y][x]) == 10:
                continue
            if any(int(resolved_grid[ny][nx]) == 10 for nx, ny in _neighbor_coordinates(x, y, width, height)):
                coastal[y, x] = 1.0
    return coastal


def _distance_map_to_targets(
    resolved_grid: list[list[int]],
    target_mask: np.ndarray,
) -> np.ndarray:
    height = len(resolved_grid)
    width = len(resolved_grid[0]) if height else 0
    coordinates = np.argwhere(target_mask > 0)
    if coordinates.size == 0:
        return np.full((height, width), float(height + width), dtype=np.float32)

    y_grid, x_grid = np.indices((height, width), dtype=np.int16)
    delta_y = np.abs(coordinates[:, 0][:, None, None] - y_grid[None, :, :])
    delta_x = np.abs(coordinates[:, 1][:, None, None] - x_grid[None, :, :])
    return np.min(delta_y + delta_x, axis=0).astype(np.float32)


def build_heuristic_prediction(
    initial_grid: list[list[int]],
    settlements: list[dict[str, Any]] | None = None,
    *,
    floor: float = MIN_PROBABILITY,
) -> np.ndarray:
    resolved_grid = overlay_initial_settlements(initial_grid, settlements)
    base = build_baseline_prediction(initial_grid, settlements, floor=floor).astype(np.float64)
    height = len(resolved_grid)
    width = len(resolved_grid[0]) if height else 0

    coastal = _coastal_mask(resolved_grid)
    settlement_mask = np.isin(np.asarray(resolved_grid, dtype=np.int16), [1, 2]).astype(np.float32)
    port_mask = (np.asarray(resolved_grid, dtype=np.int16) == 2).astype(np.float32)
    forest_mask = (np.asarray(resolved_grid, dtype=np.int16) == 4).astype(np.float32)

    settlement_distance = _distance_map_to_targets(resolved_grid, settlement_mask)
    port_distance = _distance_map_to_targets(resolved_grid, port_mask)
    forest_distance = _distance_map_to_targets(resolved_grid, forest_mask)

    settlement_influence = np.clip((MAX_INFLUENCE_DISTANCE - settlement_distance) / MAX_INFLUENCE_DISTANCE, 0.0, 1.0)
    port_influence = np.clip((MAX_INFLUENCE_DISTANCE - port_distance) / MAX_INFLUENCE_DISTANCE, 0.0, 1.0)
    forest_influence = np.clip((MAX_INFLUENCE_DISTANCE - forest_distance) / MAX_INFLUENCE_DISTANCE, 0.0, 1.0)

    for y in range(height):
        for x in range(width):
            cell = int(resolved_grid[y][x])
            influence = settlement_influence[y, x]
            port_bias = port_influence[y, x] * coastal[y, x]
            forest_bias = forest_influence[y, x]

            if cell in {10, 5}:
                continue

            adjustment = np.zeros(CLASS_COUNT, dtype=np.float64)
            if cell == 11:
                adjustment += np.asarray(
                    [
                        -0.18 * influence,
                        0.16 * influence,
                        0.05 * port_bias,
                        0.03 * (influence * (1.0 - coastal[y, x])),
                        -0.04 * influence,
                        0.0,
                    ]
                )
            elif cell == 4:
                adjustment += np.asarray(
                    [
                        -0.08 * influence,
                        0.12 * influence,
                        0.02 * port_bias,
                        0.03 * (0.6 * influence),
                        -0.10 * influence,
                        0.0,
                    ]
                )
            elif cell == 1:
                adjustment += np.asarray(
                    [
                        0.10 * (1.0 - influence),
                        -0.04 * (1.0 - influence),
                        0.05 * port_bias,
                        0.08 * (1.0 - influence * 0.5),
                        -0.02,
                        0.0,
                    ]
                )
            elif cell == 2:
                adjustment += np.asarray(
                    [
                        0.06 * (1.0 - port_bias),
                        0.04 * (1.0 - port_bias),
                        -0.08 * (1.0 - port_bias),
                        0.06 * (1.0 - port_bias),
                        -0.02,
                        0.0,
                    ]
                )
            elif cell == 3:
                adjustment += np.asarray(
                    [
                        0.05 * (1.0 - influence),
                        0.06 * influence,
                        0.01 * port_bias,
                        -0.08 * influence,
                        0.04 * (1.0 - influence),
                        0.0,
                    ]
                )
            else:
                adjustment += np.asarray(
                    [
                        -0.10 * influence,
                        0.08 * influence,
                        0.03 * port_bias,
                        0.02 * influence,
                        -0.02 * influence,
                        0.0,
                    ]
                )

            if coastal[y, x] > 0:
                adjustment[2] += 0.03 * settlement_influence[y, x]
                adjustment[0] -= 0.02 * settlement_influence[y, x]

            if cell != 4 and forest_bias > 0 and influence < 0.25:
                adjustment[4] += 0.04 * forest_bias
                adjustment[0] -= 0.02 * forest_bias

            base[y, x] += adjustment

    return normalize_prediction(base, floor=floor)


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


def normalize_prediction(prediction: np.ndarray, *, floor: float = MIN_PROBABILITY) -> np.ndarray:
    clipped = np.maximum(prediction, 0.0)
    total = clipped.sum(axis=-1, keepdims=True)
    normalized = np.where(total > 0, clipped / total, 1.0 / prediction.shape[-1])
    # Enforce floor: pin small values to floor, scale the rest down to keep sum=1
    for _ in range(5):
        below = normalized < floor
        if not below.any():
            break
        above = ~below
        n_below = below.sum(axis=-1, keepdims=True)
        above_sum = (normalized * above).sum(axis=-1, keepdims=True)
        target = 1.0 - n_below * floor
        scale = np.where(above_sum > 0, target / above_sum, 1.0)
        normalized = np.where(below, floor, normalized * scale)
    return normalized


@dataclass(slots=True)
class SeedPredictor:
    counts: np.ndarray
    floor: float = MIN_PROBABILITY

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

    def observe(self, observation: dict[str, Any]) -> None:
        viewport = observation["viewport"]
        x0 = int(viewport["x"])
        y0 = int(viewport["y"])
        grid = observation["grid"]

        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                class_index = terrain_code_to_class_index(int(cell))
                self.counts[y0 + dy, x0 + dx, class_index] += 1.0

    def prediction(self) -> np.ndarray:
        return normalize_prediction(self.counts.copy(), floor=self.floor)
