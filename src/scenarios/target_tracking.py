"""Target tracking scenario - cooperative tracking of moving targets."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class TargetTrackingScenario(BaseScenario):
    """Track multiple moving targets cooperatively."""

    def get_name(self) -> str:
        return "target_tracking"

    def setup(self):
        """Setup moving targets and obstacles."""
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=15,
            num_dynamic=0
        )

        num_targets = self.scenario_config.get('num_targets', 3)
        target_velocity = self.scenario_config.get('target_velocity', 5.0)
        self.tracking_radius = self.scenario_config.get('tracking_radius', 10.0)

        # Generate targets with random velocities
        self.targets = []
        for _ in range(num_targets):
            position = np.random.uniform(
                [50, 50, 20],
                [150, 150, 50],
                3
            )
            velocity = np.random.randn(3)
            velocity = velocity / np.linalg.norm(velocity) * target_velocity
            velocity[2] *= 0.3  # Less vertical movement

            self.environment.add_target(position, velocity)
            self.targets.append(position)

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate tracking performance using weighted-sum fitness.

        Combines:
        - Distance to targets (minimize)
        - Target coverage (ensure all targets tracked)
        - Over-assignment penalty (avoid redundancy)
        - Inter-UAV separation (collision avoidance)

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)
        targets = self.environment.get_targets()

        if len(targets) == 0:
            return 1000.0

        fitness = 0.0

        # For each target, find closest UAV
        target_tracking_distances = []
        uavs_per_target = []

        for target in targets:
            distances_to_target = np.linalg.norm(positions - target, axis=1)
            min_distance = np.min(distances_to_target)
            target_tracking_distances.append(min_distance)

            # Count UAVs within tracking radius
            tracking_uavs = np.sum(distances_to_target <= self.tracking_radius)
            uavs_per_target.append(tracking_uavs)

        # Penalty for poor tracking (distance to targets)
        fitness += np.mean(target_tracking_distances) * 2.0

        # Penalty for untracked targets
        for num_tracking in uavs_per_target:
            if num_tracking == 0:
                fitness += 100.0  # Heavy penalty

        # Penalty for over-assignment (multiple UAVs on same target)
        optimal_uavs_per_target = self.num_uavs / len(targets)
        for num_tracking in uavs_per_target:
            if num_tracking > optimal_uavs_per_target:
                fitness += (num_tracking - optimal_uavs_per_target) * 10.0

        # Inter-UAV separation
        min_separation = 3.0
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_separation:
                    fitness += (min_separation - distance) ** 2 * 5.0

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
