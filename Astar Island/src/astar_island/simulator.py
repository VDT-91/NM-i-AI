from __future__ import annotations

from dataclasses import dataclass
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
