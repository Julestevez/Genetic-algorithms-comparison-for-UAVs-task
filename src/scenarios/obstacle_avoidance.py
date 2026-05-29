"""Obstacle avoidance scenario - navigate through complex obstacle field."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class ObstacleAvoidanceScenario(BaseScenario):
    """Navigate from start to goal while avoiding obstacles."""

    def get_name(self) -> str:
        return "obstacle_avoidance"

    def setup(self):
        """Setup obstacles and start/goal positions."""
        num_obstacles = self.scenario_config.get('num_obstacles', 30)
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=num_obstacles,
            num_dynamic=0
        )

        # Define start and goal zones
        self.start_zone = np.array(self.scenario_config.get('start_zone', [10, 10, 10]))
        self.goal_zone = np.array(self.scenario_config.get('goal_zone', [190, 190, 30]))

    def get_initial_positions(self) -> np.ndarray:
        """Start UAVs near start zone."""
        positions = np.random.normal(
            self.start_zone,
            5.0,
            (self.num_uavs, 3)
        )
        # Ensure within bounds
        positions = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
        return positions

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate fitness using weighted-sum of multiple objectives.

        Combines:
        - goal_distance: Average distance to goal (weight: 1.0)
        - obstacle_proximity: Penalty for being near obstacles (weight: 10.0)
        - collision_risk: Inter-UAV collision penalty (weight: 5.0)
        - spread: Swarm cohesion penalty (weight: 0.1)

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)

        # 1. Goal distance (minimize)
        distances_to_goal = np.linalg.norm(positions - self.goal_zone, axis=1)
        goal_distance = float(np.mean(distances_to_goal))

        # 2. Obstacle proximity (minimize)
        obstacle_penalty = 0.0
        safety_distance = 5.0

        for pos in positions:
            _, distance = self.environment.obstacle_manager.get_closest_obstacle(pos)
            if distance < safety_distance:
                obstacle_penalty += (safety_distance - distance) ** 2

        # 3. Collision risk (minimize)
        min_separation = 3.0
        separation_penalty = 0.0

        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_separation:
                    separation_penalty += (min_separation - distance) ** 2

        # 4. Swarm spread (minimize - keep together)
        centroid = np.mean(positions, axis=0)
        spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

        # Weighted sum
        fitness = (
            1.0 * goal_distance +
            10.0 * obstacle_penalty +
            5.0 * separation_penalty +
            0.1 * spread
        )

        return fitness

    def evaluate_objectives(self, positions: np.ndarray) -> np.ndarray:
        """
        Evaluate TRUE multi-objective fitness (returns vector).

        Returns 4 objectives (all minimized):
        1. goal_distance: Average distance to goal
        2. obstacle_proximity: Penalty for being near obstacles
        3. collision_risk: Inter-UAV collision penalty
        4. spread: Swarm cohesion

        Args:
            positions: UAV positions array

        Returns:
            Array of 4 objective values: [goal_dist, obstacle_pen, collision_pen, spread]
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)

        # 1. Goal distance (minimize)
        distances_to_goal = np.linalg.norm(positions - self.goal_zone, axis=1)
        goal_distance = float(np.mean(distances_to_goal))

        # 2. Obstacle proximity (minimize)
        obstacle_penalty = 0.0
        safety_distance = 5.0

        for pos in positions:
            _, distance = self.environment.obstacle_manager.get_closest_obstacle(pos)
            if distance < safety_distance:
                obstacle_penalty += (safety_distance - distance) ** 2

        # 3. Collision risk (minimize)
        min_separation = 3.0
        separation_penalty = 0.0

        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < min_separation:
                    separation_penalty += (min_separation - distance) ** 2

        # 4. Swarm spread (minimize - keep together)
        centroid = np.mean(positions, axis=0)
        spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

        # Return as vector (NO weighting!)
        return np.array([goal_distance, obstacle_penalty, separation_penalty, spread])

    def get_objective_names(self) -> list:
        """Get names of the 4 objectives."""
        return ["goal_distance", "obstacle_proximity", "collision_risk", "swarm_cohesion"]
