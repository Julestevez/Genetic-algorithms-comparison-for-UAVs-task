"""Formation flight scenario - maintain desired formation while moving."""

import numpy as np
from typing import Dict
from .base_scenario import BaseScenario


class FormationFlightScenario(BaseScenario):
    """Maintain formation while moving to destination."""

    def get_name(self) -> str:
        return "formation_flight"

    def setup(self):
        """Setup formation parameters and obstacles."""
        self.environment.obstacle_manager.generate_random_obstacles(
            num_static=20,
            num_dynamic=0
        )

        self.formation_type = self.scenario_config.get('formation_type', 'v_shape')
        self.formation_scale = self.scenario_config.get('formation_scale', 10.0)
        self.leader_position = np.array([100, 100, 30])

        # Generate desired formation
        self.desired_formation = self._generate_formation()

    def _generate_formation(self) -> np.ndarray:
        """Generate desired formation pattern."""
        formations = np.zeros((self.num_uavs, 3))

        if self.formation_type == 'v_shape':
            # V-formation (leader at front)
            formations[0] = [0, 0, 0]  # Leader
            for i in range(1, self.num_uavs):
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                formations[i] = [
                    -row * self.formation_scale * 0.8,
                    side * row * self.formation_scale,
                    0
                ]

        elif self.formation_type == 'line':
            # Line formation
            for i in range(self.num_uavs):
                formations[i] = [0, i * self.formation_scale, 0]

        elif self.formation_type == 'circle':
            # Circular formation
            for i in range(self.num_uavs):
                angle = 2 * np.pi * i / self.num_uavs
                formations[i] = [
                    self.formation_scale * np.cos(angle),
                    self.formation_scale * np.sin(angle),
                    0
                ]

        elif self.formation_type == 'grid':
            # Grid formation
            cols = int(np.ceil(np.sqrt(self.num_uavs)))
            for i in range(self.num_uavs):
                row = i // cols
                col = i % cols
                formations[i] = [
                    row * self.formation_scale,
                    col * self.formation_scale,
                    0
                ]

        return formations

    def evaluate_solution(self, positions: np.ndarray) -> float:
        """
        Evaluate formation quality using weighted-sum fitness.

        Combines:
        - Formation shape maintenance
        - Centroid movement toward goal
        - Distance variance matching
        - Obstacle avoidance

        Args:
            positions: UAV positions array

        Returns:
            Weighted-sum fitness value (lower is better)
        """
        # Reshape if needed
        if positions.ndim == 1:
            positions = positions.reshape(self.num_uavs, 3)
        # Calculate formation relative to leader/centroid
        centroid = np.mean(positions, axis=0)

        # Calculate current formation relative positions
        relative_positions = positions - centroid

        # Calculate desired positions
        desired_positions = self.desired_formation + self.leader_position

        # Formation error (how well does current formation match desired)
        formation_error = 0.0
        for i in range(self.num_uavs):
            # Find best matching desired position (for any formation rotation)
            errors = np.linalg.norm(desired_positions - positions[i], axis=1)
            min_error = np.min(errors)
            formation_error += min_error

        fitness = formation_error * 2.0

        # Distance of centroid to leader target position
        centroid_error = np.linalg.norm(centroid - self.leader_position)
        fitness += centroid_error * 1.0

        # Inter-UAV distance variance (should match formation distances)
        current_distances = []
        desired_distances = []
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                current_distances.append(np.linalg.norm(positions[i] - positions[j]))
                desired_distances.append(np.linalg.norm(
                    self.desired_formation[i] - self.desired_formation[j]
                ))

        if len(current_distances) > 0:
            distance_variance = np.var(np.array(current_distances) - np.array(desired_distances))
            fitness += distance_variance * 0.5

        # Obstacle avoidance
        obstacle_penalty = 0.0
        safety_distance = 5.0
        for pos in positions:
            _, distance = self.environment.obstacle_manager.get_closest_obstacle(pos)
            if distance < safety_distance:
                obstacle_penalty += (safety_distance - distance) ** 2

        fitness += obstacle_penalty * 10.0

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
