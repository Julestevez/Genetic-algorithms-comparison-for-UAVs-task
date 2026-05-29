"""Base class for all benchmark scenarios."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from ..environment.environment import Environment
from ..simulation.swarm import Swarm


class BaseScenario(ABC):
    """Abstract base class for benchmark scenarios."""

    def __init__(self, config: Dict, num_uavs: int):
        """
        Initialize scenario.

        Args:
            config: Configuration dictionary
            num_uavs: Number of UAVs
        """
        self.config = config
        self.num_uavs = num_uavs
        self.scenario_config = config.get('scenarios', {}).get(self.get_name(), {})

        # Environment and swarm
        self.environment = Environment(config)
        self.swarm = Swarm(num_uavs, config)

        # Bounds for optimization
        env_bounds = config.get('environment', {}).get('bounds', {})
        self.bounds = np.array([
            [env_bounds.get('x_min', 0), env_bounds.get('x_max', 200)],
            [env_bounds.get('y_min', 0), env_bounds.get('y_max', 200)],
            [env_bounds.get('z_min', 0), env_bounds.get('z_max', 100)]
        ])

    @abstractmethod
    def get_name(self) -> str:
        """Get scenario name."""
        pass

    @abstractmethod
    def setup(self):
        """Setup scenario-specific environment and objectives."""
        pass

    @abstractmethod
    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate fitness function for single-objective optimization.

        Returns a scalar fitness value (weighted-sum of objectives).
        Used by single-objective algorithms: PSO, GWO, ACO.

        Args:
            positions: Flattened array or (num_uavs, 3) array

        Returns:
            Fitness value (lower is better)
        """
        pass

    @abstractmethod
    def evaluate_objectives(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate objectives for TRUE multi-objective optimization.

        Returns a vector of objective values (NOT weighted sum).
        Used by multi-objective algorithms: MOPSO, MOGWO, MOACO.

        Args:
            positions: Flattened array or (num_uavs, 3) array

        Returns:
            Array of objective values, shape (n_objectives,), all minimized
            Example: np.array([path_length, collisions, energy, time])
        """
        pass

    def get_objective_names(self) -> list:
        """
        Get names of objectives for visualization and metrics.

        Returns:
            List of objective names
        """
        return ["fitness"]  # Default for scenarios with single objective

    def get_initial_positions(self) -> np.ndarray:
        """Get initial UAV positions."""
        # Default: random positions in start zone
        positions = np.random.uniform(
            self.bounds[:, 0].reshape(1, -1),
            self.bounds[:, 1].reshape(1, -1),
            (self.num_uavs, 3)
        )
        return positions

    def get_bounds(self) -> np.ndarray:
        """Get search space bounds."""
        return self.bounds.copy()

    def reset(self):
        """Reset scenario."""
        self.environment.reset()
        self.swarm.reset()
        self.setup()
