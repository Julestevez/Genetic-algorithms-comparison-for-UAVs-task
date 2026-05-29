"""Area coverage scenario - maximize surveillance coverage."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class AreaCoverageScenario(BaseScenario):
    """Maximize coverage of specified area."""

    def get_name(self) -> str:
        return "area_coverage"

    def setup(self):
        """Setup coverage area and minimal obstacles."""
        # Fewer obstacles for coverage scenario
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=10,
            num_dynamic=0
        )

        coverage_area = self.scenario_config.get('coverage_area', [0, 200, 0, 200])
        self.coverage_x_min, self.coverage_x_max = coverage_area[0], coverage_area[1]
        self.coverage_y_min, self.coverage_y_max = coverage_area[2], coverage_area[3]

        self.sensor_range = self.config.get('uav', {}).get('sensor_range', 50.0)
        self.grid_resolution = self.scenario_config.get('grid_resolution', 5.0)

        # Initialize coverage tracking
        self.coverage_grid = np.zeros((
            int((self.coverage_x_max - self.coverage_x_min) / self.grid_resolution),
            int((self.coverage_y_max - self.coverage_y_min) / self.grid_resolution)
        ))

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate coverage quality using weighted-sum fitness.
        Lower fitness = better coverage.

        Combines:
        - Coverage ratio (maximize covered area)
        - Overlap penalty (minimize inefficient coverage)
        - Distribution uniformity (encourage spread)

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)
        # Reset coverage grid
        self.coverage_grid.fill(0)

        # Mark covered cells
        for pos in positions:
            x_idx = int((pos[0] - self.coverage_x_min) / self.grid_resolution)
            y_idx = int((pos[1] - self.coverage_y_min) / self.grid_resolution)

            radius_cells = int(self.sensor_range / self.grid_resolution)

            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    nx, ny = x_idx + dx, y_idx + dy

                    if (0 <= nx < self.coverage_grid.shape[0] and
                        0 <= ny < self.coverage_grid.shape[1]):

                        cell_pos = np.array([
                            self.coverage_x_min + nx * self.grid_resolution,
                            self.coverage_y_min + ny * self.grid_resolution
                        ])

                        if np.linalg.norm(cell_pos - pos[:2]) <= self.sensor_range:
                            self.coverage_grid[nx, ny] += 1

        # Calculate coverage metrics
        total_cells = self.coverage_grid.size
        covered_cells = np.sum(self.coverage_grid > 0)
        coverage_ratio = covered_cells / total_cells

        # Penalty for uncovered area
        fitness = (1.0 - coverage_ratio) * 1000.0

        # Penalty for overlap (inefficient coverage)
        overlap_penalty = np.sum(self.coverage_grid > 1) / total_cells
        fitness += overlap_penalty * 100.0

        # Encourage uniform distribution
        centroid = np.mean(positions, axis=0)
        spread_variance = np.var(np.linalg.norm(positions - centroid, axis=1))
        fitness += abs(spread_variance - 400.0) * 0.5  # Target spread ~20m std

        return fitness

    def evaluate_objectives(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate TRUE multi-objective fitness (returns vector).

        For this scenario, returns single objective (can be expanded later).

        Args:
            positions: UAV positions array

        Returns:
            Array of objective values
        """
        fitness = self.evaluate_solution(positions)
        return np.array([fitness])

    def get_objective_names(self) -> list:
        """Get names of objectives."""
        return ["fitness"]
