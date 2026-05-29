"""Grey Wolf Optimizer implementation for UAV swarm."""

import numpy as np
from typing import Dict, Any
from .base_optimizer import BaseOptimizer


class GWO(BaseOptimizer):
    """Grey Wolf Optimizer algorithm."""

    def __init__(self, config: Dict[str, Any], num_uavs: int):
        """Initialize GWO."""
        super().__init__(config, num_uavs)

        gwo_config = config.get('algorithms', {}).get('gwo', {})
        self.pack_size = gwo_config.get('pack_size', 30)
        self.a_decay = gwo_config.get('a_decay', 2.0)

        # GWO specific variables
        self.positions = None
        self.alpha_position = None  # Best solution
        self.beta_position = None   # Second best
        self.delta_position = None  # Third best
        self.alpha_fitness = float('inf')
        self.beta_fitness = float('inf')
        self.delta_fitness = float('inf')
        self.bounds = None
        self.current_iteration = 0
        self.max_iterations = 100

    def initialize(self, bounds: np.ndarray):
        """Initialize wolf pack."""
        self.bounds = bounds
        position_shape = (self.pack_size, self.num_uavs, self.dimension)

        # Initialize positions randomly within bounds
        self.positions = np.random.uniform(
            bounds[:, 0].reshape(1, 1, -1),
            bounds[:, 1].reshape(1, 1, -1),
            position_shape
        )

        # Initialize alpha, beta, delta
        self.alpha_position = self.positions[0].copy()
        self.beta_position = self.positions[1].copy()
        self.delta_position = self.positions[2].copy()
        self.current_iteration = 0

    def update(self, fitness_values: np.ndarray):
        """Update wolf positions based on alpha, beta, delta."""
        # Update alpha, beta, delta
        for i in range(self.pack_size):
            if fitness_values[i] < self.alpha_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position.copy()
                self.beta_fitness = self.alpha_fitness
                self.beta_position = self.alpha_position.copy()
                self.alpha_fitness = fitness_values[i]
                self.alpha_position = self.positions[i].copy()
            elif fitness_values[i] < self.beta_fitness:
                self.delta_fitness = self.beta_fitness
                self.delta_position = self.beta_position.copy()
                self.beta_fitness = fitness_values[i]
                self.beta_position = self.positions[i].copy()
            elif fitness_values[i] < self.delta_fitness:
                self.delta_fitness = fitness_values[i]
                self.delta_position = self.positions[i].copy()

        # Update global best
        self.global_best_position = self.alpha_position.copy()
        self.global_best_fitness = self.alpha_fitness

        # Linearly decrease a from 2 to 0
        a = self.a_decay * (1 - self.current_iteration / self.max_iterations)

        # Update positions
        for i in range(self.pack_size):
            for j in range(self.num_uavs):
                for k in range(self.dimension):
                    # Calculate distance to alpha, beta, delta
                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_position[j, k] - self.positions[i, j, k])
                    X1 = self.alpha_position[j, k] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_position[j, k] - self.positions[i, j, k])
                    X2 = self.beta_position[j, k] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_position[j, k] - self.positions[i, j, k])
                    X3 = self.delta_position[j, k] - A3 * D_delta

                    # Update position
                    self.positions[i, j, k] = (X1 + X2 + X3) / 3.0

        # Apply boundary constraints
        self.positions = np.clip(
            self.positions,
            self.bounds[:, 0].reshape(1, 1, -1),
            self.bounds[:, 1].reshape(1, 1, -1)
        )

        self.current_iteration += 1

    def get_positions(self) -> np.ndarray:
        """Get current wolf positions."""
        return self.positions.copy()

    def optimize(self, objective_function, bounds, max_iterations):
        """Override to set max_iterations for a parameter."""
        self.max_iterations = max_iterations
        return super().optimize(objective_function, bounds, max_iterations)
