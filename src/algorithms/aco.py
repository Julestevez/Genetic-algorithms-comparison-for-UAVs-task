"""Ant Colony Optimization implementation for UAV swarm."""

import numpy as np
from typing import Dict, Any
from .base_optimizer import BaseOptimizer


class ACO(BaseOptimizer):
    """Ant Colony Optimization algorithm adapted for continuous optimization."""

    def __init__(self, config: Dict[str, Any], num_uavs: int):
        """Initialize ACO."""
        super().__init__(config, num_uavs)

        aco_config = config.get('algorithms', {}).get('aco', {})
        self.colony_size = aco_config.get('colony_size', 30)
        self.alpha = aco_config.get('alpha', 1.0)  # pheromone importance
        self.beta = aco_config.get('beta', 2.0)    # heuristic importance
        self.rho = aco_config.get('rho', 0.1)      # evaporation rate
        self.q = aco_config.get('q', 100)          # pheromone deposit

        # ACO specific variables
        self.positions = None
        self.pheromone = None
        self.bounds = None
        self.archive_size = 10
        self.solution_archive = []
        self.fitness_archive = []

    def initialize(self, bounds: np.ndarray):
        """Initialize ant colony."""
        self.bounds = bounds
        position_shape = (self.colony_size, self.num_uavs, self.dimension)

        # Initialize positions randomly
        self.positions = np.random.uniform(
            bounds[:, 0].reshape(1, 1, -1),
            bounds[:, 1].reshape(1, 1, -1),
            position_shape
        )

        # Initialize pheromone matrix (simplified grid-based approach)
        self.grid_size = 20
        self.pheromone = np.ones((
            self.num_uavs,
            self.dimension,
            self.grid_size
        )) * 0.1

        self.solution_archive = []
        self.fitness_archive = []

    def update(self, fitness_values: np.ndarray):
        """Update pheromone trails and ant positions."""
        # Update solution archive
        for i in range(self.colony_size):
            if len(self.solution_archive) < self.archive_size:
                self.solution_archive.append(self.positions[i].copy())
                self.fitness_archive.append(fitness_values[i])
            else:
                # Replace worst solution if current is better
                worst_idx = np.argmax(self.fitness_archive)
                if fitness_values[i] < self.fitness_archive[worst_idx]:
                    self.solution_archive[worst_idx] = self.positions[i].copy()
                    self.fitness_archive[worst_idx] = fitness_values[i]

        # Update global best
        best_idx = np.argmin(fitness_values)
        if fitness_values[best_idx] < self.global_best_fitness:
            self.global_best_fitness = fitness_values[best_idx]
            self.global_best_position = self.positions[best_idx].copy()

        # Evaporate pheromone
        self.pheromone *= (1 - self.rho)

        # Deposit pheromone based on solution quality
        for pos, fit in zip(self.solution_archive, self.fitness_archive):
            pheromone_amount = self.q / (fit + 1e-10)
            self._deposit_pheromone(pos, pheromone_amount)

        # Construct new solutions based on pheromone
        self._construct_solutions()

    def _deposit_pheromone(self, position: np.ndarray, amount: float):
        """Deposit pheromone at given position."""
        for uav_idx in range(self.num_uavs):
            for dim in range(self.dimension):
                # Normalize position to grid
                normalized = (position[uav_idx, dim] - self.bounds[dim, 0]) / \
                           (self.bounds[dim, 1] - self.bounds[dim, 0])
                grid_idx = int(normalized * (self.grid_size - 1))
                grid_idx = np.clip(grid_idx, 0, self.grid_size - 1)

                self.pheromone[uav_idx, dim, grid_idx] += amount

    def _construct_solutions(self):
        """Construct new solutions based on pheromone trails."""
        for ant in range(self.colony_size):
            for uav_idx in range(self.num_uavs):
                for dim in range(self.dimension):
                    # Probability distribution based on pheromone
                    pheromone_trail = self.pheromone[uav_idx, dim, :]

                    # Add heuristic information (attraction to best solutions)
                    heuristic = np.ones(self.grid_size)
                    if len(self.solution_archive) > 0:
                        best_solution = self.solution_archive[np.argmin(self.fitness_archive)]
                        best_pos_normalized = (best_solution[uav_idx, dim] - self.bounds[dim, 0]) / \
                                            (self.bounds[dim, 1] - self.bounds[dim, 0])
                        best_grid_idx = int(best_pos_normalized * (self.grid_size - 1))

                        # Higher heuristic near best solution
                        for i in range(self.grid_size):
                            distance = abs(i - best_grid_idx) / self.grid_size
                            heuristic[i] = 1.0 / (distance + 0.1)

                    # Calculate probabilities
                    probabilities = (pheromone_trail ** self.alpha) * (heuristic ** self.beta)
                    probabilities /= probabilities.sum()

                    # Select grid cell
                    grid_idx = np.random.choice(self.grid_size, p=probabilities)

                    # Convert back to position with some randomness
                    normalized_pos = (grid_idx + np.random.rand()) / self.grid_size
                    self.positions[ant, uav_idx, dim] = \
                        self.bounds[dim, 0] + normalized_pos * (self.bounds[dim, 1] - self.bounds[dim, 0])

        # Apply boundary constraints
        self.positions = np.clip(
            self.positions,
            self.bounds[:, 0].reshape(1, 1, -1),
            self.bounds[:, 1].reshape(1, 1, -1)
        )

    def get_positions(self) -> np.ndarray:
        """Get current ant positions."""
        return self.positions.copy()
