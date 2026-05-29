"""Particle Swarm Optimization implementation for UAV swarm."""

import numpy as np
from typing import Dict, Any
from .base_optimizer import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimization algorithm."""

    def __init__(self, config: Dict[str, Any], num_uavs: int):
        """Initialize PSO."""
        super().__init__(config, num_uavs)

        pso_config = config.get('algorithms', {}).get('pso', {})
        self.swarm_size = pso_config.get('swarm_size', 30)
        self.w = pso_config.get('w', 0.7)  # inertia weight
        self.c1 = pso_config.get('c1', 1.5)  # cognitive parameter
        self.c2 = pso_config.get('c2', 1.5)  # social parameter
        self.w_damp = pso_config.get('w_damp', 0.99)

        # PSO specific variables
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_fitness = None
        self.bounds = None

    def initialize(self, bounds: np.ndarray):
        """Initialize particle swarm."""
        self.bounds = bounds
        position_shape = (self.swarm_size, self.num_uavs, self.dimension)

        # Initialize positions randomly within bounds
        self.positions = np.random.uniform(
            bounds[:, 0].reshape(1, 1, -1),
            bounds[:, 1].reshape(1, 1, -1),
            position_shape
        )

        # Initialize velocities
        velocity_range = (bounds[:, 1] - bounds[:, 0]) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range.reshape(1, 1, -1),
            velocity_range.reshape(1, 1, -1),
            position_shape
        )

        # Initialize personal bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(self.swarm_size, float('inf'))

    def update(self, fitness_values: np.ndarray):
        """Update particle positions and velocities."""
        # Update personal bests
        improved = fitness_values < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness_values[improved]
        self.personal_best_positions[improved] = self.positions[improved].copy()

        # Update global best
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best_positions[best_idx].copy()

        # Update velocities
        r1 = np.random.rand(self.swarm_size, self.num_uavs, self.dimension)
        r2 = np.random.rand(self.swarm_size, self.num_uavs, self.dimension)

        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)

        self.velocities = self.w * self.velocities + cognitive + social

        # Limit velocities
        max_velocity = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.2
        self.velocities = np.clip(
            self.velocities,
            -max_velocity.reshape(1, 1, -1),
            max_velocity.reshape(1, 1, -1)
        )

        # Update positions
        self.positions = self.positions + self.velocities

        # Apply boundary constraints
        self.positions = np.clip(
            self.positions,
            self.bounds[:, 0].reshape(1, 1, -1),
            self.bounds[:, 1].reshape(1, 1, -1)
        )

        # Damp inertia weight
        self.w *= self.w_damp

    def get_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.positions.copy()
