"""Base class for all optimization algorithms."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable, Any
import time


class BaseOptimizer(ABC):
    """Abstract base class for bio-inspired optimization algorithms."""

    def __init__(self, config: Dict[str, Any], num_uavs: int):
        """
        Initialize optimizer.

        Args:
            config: Configuration dictionary
            num_uavs: Number of UAVs in the swarm
        """
        self.config = config
        self.num_uavs = num_uavs
        self.dimension = 3  # 3D space (x, y, z)

        # Convergence tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.iteration_times = []

        # Best solution tracking
        self.global_best_position = None
        self.global_best_fitness = float('inf')

        # Statistics
        self.num_evaluations = 0
        self.start_time = None
        self.end_time = None

    @abstractmethod
    def initialize(self, bounds: np.ndarray):
        """
        Initialize the optimizer population.

        Args:
            bounds: Array of shape (dimension, 2) with [min, max] for each dimension
        """
        pass

    @abstractmethod
    def update(self, fitness_values: np.ndarray):
        """
        Update the optimizer state based on fitness values.

        Args:
            fitness_values: Array of fitness values for current population
        """
        pass

    @abstractmethod
    def get_positions(self) -> np.ndarray:
        """
        Get current positions of all agents in the population.

        Returns:
            Array of shape (population_size, num_uavs, dimension)
        """
        pass

    def optimize(self,
                 objective_function: Callable,
                 bounds: np.ndarray,
                 max_iterations: int) -> Tuple[np.ndarray, float, Dict]:
        """
        Run the optimization process.

        Args:
            objective_function: Function to minimize
            bounds: Search space bounds
            max_iterations: Maximum number of iterations

        Returns:
            Tuple of (best_position, best_fitness, statistics)
        """
        self.start_time = time.time()
        self.initialize(bounds)

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Get current positions
            positions = self.get_positions()

            # Evaluate fitness for each solution
            fitness_values = np.array([
                objective_function(pos) for pos in positions
            ])
            self.num_evaluations += len(fitness_values)

            # Update global best
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < self.global_best_fitness:
                self.global_best_fitness = fitness_values[min_idx]
                self.global_best_position = positions[min_idx].copy()

            # Update algorithm state
            self.update(fitness_values)

            # Track statistics
            self.best_fitness_history.append(self.global_best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.diversity_history.append(self._calculate_diversity(positions))
            self.iteration_times.append(time.time() - iter_start)

            # Early stopping if converged
            if self._check_convergence():
                break

        self.end_time = time.time()

        statistics = self._get_statistics()
        return self.global_best_position, self.global_best_fitness, statistics

    def optimize_multi_objective(self,
                                 objective_function: Callable,
                                 bounds: np.ndarray,
                                 max_iterations: int) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """
        Run TRUE multi-objective optimization process.

        Args:
            objective_function: Function that returns np.ndarray of objectives
            bounds: Search space bounds
            max_iterations: Maximum number of iterations

        Returns:
            Tuple of (pareto_front_positions, pareto_front_objectives, statistics)
            - pareto_front_positions: List of non-dominated solution positions
            - pareto_front_objectives: List of objective vectors for each solution
            - statistics: Dict with optimization stats
        """
        self.start_time = time.time()
        self.initialize(bounds)

        for iteration in range(max_iterations):
            iter_start = time.time()

            # Get current positions
            positions = self.get_positions()

            # Evaluate objectives for each solution (returns np.ndarray per solution)
            objective_values = [objective_function(pos) for pos in positions]
            self.num_evaluations += len(objective_values)

            # Update algorithm state with objective vectors
            if hasattr(self, 'update_with_objectives'):
                self.update_with_objectives(objective_values)
            else:
                # Fallback: convert to scalar fitness (simple sum)
                fitness_values = np.array([np.sum(obj) for obj in objective_values])
                self.update(fitness_values)

            # Track statistics (use sum for compatibility)
            fitness_values = np.array([np.sum(obj) for obj in objective_values])
            self.best_fitness_history.append(self.global_best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_values))
            self.diversity_history.append(self._calculate_diversity(positions))
            self.iteration_times.append(time.time() - iter_start)

        self.end_time = time.time()

        # Get Pareto front from archive (if algorithm maintains one)
        if hasattr(self, 'archive_positions') and len(self.archive_positions) > 0:
            pareto_positions = self.archive_positions
            pareto_objectives = self.archive_objectives
        else:
            # Fallback: return best single solution
            pareto_positions = [self.global_best_position]
            # Evaluate objectives for best solution
            pareto_objectives = [objective_function(self.global_best_position)]

        statistics = self._get_statistics()
        statistics['pareto_size'] = len(pareto_positions)

        return pareto_positions, pareto_objectives, statistics

    def _calculate_diversity(self, positions: np.ndarray) -> float:
        """Calculate population diversity."""
        if len(positions) < 2:
            return 0.0

        flat_positions = positions.reshape(len(positions), -1)
        centroid = np.mean(flat_positions, axis=0)
        diversity = np.mean(np.linalg.norm(flat_positions - centroid, axis=1))
        return diversity

    def _check_convergence(self, threshold: float = 1e-6, window: int = 10) -> bool:
        """Check if algorithm has converged."""
        if len(self.best_fitness_history) < window:
            return False

        recent_fitness = self.best_fitness_history[-window:]
        fitness_std = np.std(recent_fitness)
        return fitness_std < threshold

    def _get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'algorithm': self.__class__.__name__,
            'best_fitness': self.global_best_fitness,
            'num_evaluations': self.num_evaluations,
            'total_time': self.end_time - self.start_time if self.end_time else 0,
            'avg_iteration_time': np.mean(self.iteration_times) if self.iteration_times else 0,
            'convergence_iteration': self._find_convergence_iteration(),
            'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'best_fitness_history': self.best_fitness_history.copy(),
            'avg_fitness_history': self.avg_fitness_history.copy(),
            'diversity_history': self.diversity_history.copy(),
        }

    def _find_convergence_iteration(self) -> int:
        """Find iteration where algorithm converged."""
        if len(self.best_fitness_history) < 2:
            return 0

        threshold = 0.01 * abs(self.best_fitness_history[0] - self.global_best_fitness)
        for i, fitness in enumerate(self.best_fitness_history):
            if abs(fitness - self.global_best_fitness) < threshold:
                return i
        return len(self.best_fitness_history) - 1

    def reset(self):
        """Reset optimizer state."""
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.iteration_times = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.num_evaluations = 0
        self.start_time = None
        self.end_time = None
