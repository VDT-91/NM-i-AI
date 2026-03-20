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
        raw = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
    elif terrain_code == 10:  # Ocean - never changes
        raw = np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    elif terrain_code == 1:  # Settlement
        if dynamics.is_collapse:
            raw = np.array([0.55, 0.03, 0.01, 0.06, 0.33, 0.02])
        elif dynamics.is_decline:
            raw = np.array([0.30, 0.15, 0.03, 0.20, 0.28, 0.04])
        else:
            survive = 0.10 + 0.45 * sf
            ruin = 0.05 + 0.20 * (1.0 - sf)
            empty = 0.05 + 0.25 * (1.0 - sf)
            forest = 0.05 + 0.10 * (1.0 - sf)
            raw = np.array([empty, survive, 0.05, ruin, forest, 0.02])
    elif terrain_code == 2:  # Port
        if dynamics.is_collapse:
            raw = np.array([0.50, 0.02, 0.03, 0.08, 0.35, 0.02])
        elif dynamics.is_decline:
            raw = np.array([0.25, 0.08, 0.15, 0.22, 0.26, 0.04])
        else:
            survive_port = 0.10 + 0.40 * sf
            raw = np.array([0.05, 0.08, survive_port, 0.15, 0.10, 0.02])
    elif terrain_code == 3:  # Ruin
        if dynamics.is_collapse:
            raw = np.array([0.50, 0.01, 0.01, 0.05, 0.41, 0.02])
        else:
            rebuild = 0.02 + 0.15 * sf
            raw = np.array([0.15, rebuild, 0.02, 0.40, 0.30, 0.02])
    elif terrain_code == 4:  # Forest
        if dynamics.is_collapse:
            raw = np.array([0.04, 0.01, 0.01, 0.01, 0.92, 0.01])
        else:
            raw = np.array([0.05, 0.02 * sf, 0.01, 0.02, 0.88, 0.02])
    elif terrain_code == 11:  # Plains
        if dynamics.is_collapse:
            raw = np.array([0.75, 0.02, 0.01, 0.02, 0.18, 0.02])
        elif dynamics.is_decline:
            raw = np.array([0.60, 0.06, 0.02, 0.04, 0.24, 0.04])
        else:
            settle = 0.05 + 0.15 * sf
            port = 0.02 + 0.05 * sf if is_coastal else 0.01
            raw = np.array([0.55 - settle, settle, port, 0.04, 0.15, 0.03])
    else:  # Empty/other
        if dynamics.is_collapse:
            raw = np.array([0.80, 0.01, 0.01, 0.01, 0.15, 0.02])
        else:
            raw = np.array([0.70, 0.05, 0.03, 0.03, 0.14, 0.05])

    return raw / raw.sum()
