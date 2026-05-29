"""Dynamic obstacle avoidance scenario - navigate through moving obstacles."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class DynamicObstacleAvoidanceScenario(BaseScenario):
    """Navigate to goal while avoiding dynamic obstacles."""

    def get_name(self) -> str:
        return "dynamic_obstacle_avoidance"

    def setup(self):
        """Setup dynamic obstacles and navigation goals."""
        num_dynamic = self.scenario_config.get('num_dynamic_obstacles', 15)
        velocity_range = self.scenario_config.get('obstacle_velocity_range', [2.0, 8.0])

        # Generate mostly dynamic obstacles
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=5,
            num_dynamic=num_dynamic
        )

        # Override dynamic obstacle velocities with custom range
        for obstacle in self.environment.obstacle_manager.obstacles:
            if obstacle.obstacle_type == 'dynamic':
                velocity = np.random.randn(3)
                speed = np.random.uniform(velocity_range[0], velocity_range[1])
                velocity = velocity / np.linalg.norm(velocity) * speed
                velocity[2] *= 0.5  # Less vertical movement
                obstacle.velocity = velocity

        self.start_zone = np.array([20, 20, 20])
        self.goal_zone = np.array([180, 180, 30])

        self.prediction_enabled = self.scenario_config.get('prediction_enabled', True)
        self.prediction_horizon = 3.0  # seconds

    def _predict_obstacle_position(self, obstacle, time_ahead: float) -> np.ndarray:
        """Predict obstacle position at future time."""
        if obstacle.velocity is None:
            return obstacle.position.copy()

        future_position = obstacle.position + obstacle.velocity * time_ahead
        return future_position

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate dynamic avoidance using weighted-sum fitness.

        Combines:
        - Progress toward goal
        - Current obstacle avoidance
        - Predictive obstacle avoidance
        - Inter-UAV collision avoidance
        - Formation cohesion

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)
        fitness = 0.0

        # Distance to goal
        distances_to_goal = np.linalg.norm(positions - self.goal_zone, axis=1)
        fitness += np.mean(distances_to_goal) * 1.0

        # Dynamic obstacle avoidance (current positions)
        safety_distance = 6.0
        for pos in positions:
            for obstacle in self.environment.obstacle_manager.obstacles:
                distance = obstacle.distance_to_point(pos)

                if distance < safety_distance:
                    fitness += (safety_distance - distance) ** 2 * 15.0

                # Predictive avoidance
                if self.prediction_enabled and obstacle.velocity is not None:
                    # Check predicted positions
                    for t in np.linspace(0.5, self.prediction_horizon, 3):
                        future_pos = self._predict_obstacle_position(obstacle, t)
                        future_distance = np.linalg.norm(pos - future_pos)

                        if future_distance < safety_distance:
                            # Penalty decreases with time horizon
                            time_weight = 1.0 / (1.0 + t)
                            fitness += (safety_distance - future_distance) ** 2 * 5.0 * time_weight

        # Collision risk between UAVs
        min_separation = 4.0
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_separation:
                    fitness += (min_separation - distance) ** 2 * 8.0

        # Encourage forward progress (don't get stuck)
        centroid = np.mean(positions, axis=0)
        progress = np.dot(
            centroid - self.start_zone,
            self.goal_zone - self.start_zone
        ) / np.linalg.norm(self.goal_zone - self.start_zone) ** 2

        if progress < 0.1:  # Not moving forward
            fitness += 100.0

        # Formation cohesion (stay together for safety)
        spread = np.mean(np.linalg.norm(positions - centroid, axis=1))
        if spread > 30.0:  # Too spread out
            fitness += (spread - 30.0) * 2.0

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
