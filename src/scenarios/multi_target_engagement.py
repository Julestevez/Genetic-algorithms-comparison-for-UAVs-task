"""Multi-target engagement scenario - coordinate attacks on multiple targets."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class MultiTargetEngagementScenario(BaseScenario):
    """Coordinate engagement of multiple targets with priorities."""

    def get_name(self) -> str:
        return "multi_target_engagement"

    def setup(self):
        """Setup targets with different priorities."""
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=12,
            num_dynamic=0
        )

        num_targets = self.scenario_config.get('num_targets', 8)
        self.engagement_range = self.scenario_config.get('engagement_range', 15.0)

        target_types_config = self.scenario_config.get('target_types', ['high', 'medium', 'low'])

        # Generate targets with types and priorities
        self.targets = []
        self.target_types = []
        self.target_priorities = []

        for i in range(num_targets):
            position = np.random.uniform(
                [30, 30, 10],
                [170, 170, 40],
                3
            )
            target_type = np.random.choice(target_types_config)

            # Priority: high=1, medium=2, low=3
            priority = {'high': 1, 'medium': 2, 'low': 3}.get(target_type, 2)

            self.targets.append(position)
            self.target_types.append(target_type)
            self.target_priorities.append(priority)

        self.targets_engaged = [False] * num_targets
        self.coordination_required = self.scenario_config.get('coordination_required', True)
        self.min_uavs_per_high_target = 2

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate engagement strategy using weighted-sum fitness.

        Combines:
        - Target prioritization (high-value targets first)
        - Coordination quality (multiple UAVs on important targets)
        - Assignment efficiency (avoid waste)
        - Inter-UAV collision avoidance

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)
        fitness = 0.0

        # Reset engagement status
        self.targets_engaged = [False] * len(self.targets)
        uavs_per_target = [0] * len(self.targets)

        # Assign UAVs to targets
        for i, target in enumerate(self.targets):
            distances = np.linalg.norm(positions - target, axis=1)

            # Count UAVs in engagement range
            engaging_uavs = np.sum(distances <= self.engagement_range)
            uavs_per_target[i] = engaging_uavs

            if engaging_uavs > 0:
                self.targets_engaged[i] = True

            # Distance penalty (weighted by priority)
            min_distance = np.min(distances)
            priority_weight = 4.0 / self.target_priorities[i]
            fitness += min_distance * priority_weight * 0.5

        # Penalties for unengaged targets
        for i, engaged in enumerate(self.targets_engaged):
            if not engaged:
                priority_weight = 4.0 / self.target_priorities[i]
                fitness += 100.0 * priority_weight

        # Coordination requirements
        if self.coordination_required:
            for i, target_type in enumerate(self.target_types):
                if target_type == 'high':
                    # High-priority targets should have multiple UAVs
                    if uavs_per_target[i] < self.min_uavs_per_high_target:
                        shortage = self.min_uavs_per_high_target - uavs_per_target[i]
                        fitness += shortage * 50.0
                    elif uavs_per_target[i] > self.min_uavs_per_high_target + 1:
                        # Too many UAVs is also wasteful
                        excess = uavs_per_target[i] - self.min_uavs_per_high_target - 1
                        fitness += excess * 20.0

        # Efficient assignment (avoid clustering all UAVs on few targets)
        non_zero_assignments = np.sum(np.array(uavs_per_target) > 0)
        if non_zero_assignments < len(self.targets) and non_zero_assignments < self.num_uavs:
            fitness += (len(self.targets) - non_zero_assignments) * 30.0

        # Inter-UAV coordination (UAVs engaging same target should be spread)
        for i, target in enumerate(self.targets):
            engaging_uav_positions = []
            for j, pos in enumerate(positions):
                if np.linalg.norm(pos - target) <= self.engagement_range:
                    engaging_uav_positions.append(pos)

            if len(engaging_uav_positions) > 1:
                engaging_uav_positions = np.array(engaging_uav_positions)
                # They should be spread around the target
                angles = np.arctan2(
                    engaging_uav_positions[:, 1] - target[1],
                    engaging_uav_positions[:, 0] - target[0]
                )
                # Check angular separation
                for k in range(len(angles)):
                    for l in range(k + 1, len(angles)):
                        angle_diff = abs(angles[k] - angles[l])
                        if angle_diff < np.pi / 4:  # Too close angularly
                            fitness += 10.0

        # Collision avoidance
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
