"""UAV swarm management and coordination."""

import numpy as np
from typing import Dict, List, Tuple
from .uav import UAV


class Swarm:
    """Manages a swarm of UAVs."""

    def __init__(self, num_uavs: int, config: Dict):
        """
        Initialize swarm.

        Args:
            num_uavs: Number of UAVs in swarm
            config: Configuration dictionary
        """
        self.num_uavs = num_uavs
        self.config = config
        self.uavs: List[UAV] = []

        # Initialize UAVs
        for i in range(num_uavs):
            self.uavs.append(UAV(i, config))

        # Swarm metrics
        self.formation_error = 0.0
        self.connectivity = 1.0
        self.coverage_area = 0.0

    def set_positions(self, positions: np.ndarray):
        """
        Set UAV positions.

        Args:
            positions: Array of shape (num_uavs, 3)
        """
        for i, uav in enumerate(self.uavs):
            uav.state.position = positions[i].copy()
            uav.position_history = [positions[i].copy()]

    def set_targets(self, targets: np.ndarray):
        """
        Set target positions for all UAVs.

        Args:
            targets: Array of shape (num_uavs, 3)
        """
        for i, uav in enumerate(self.uavs):
            uav.set_target(targets[i])

    def update(self, dt: float, wind_force: np.ndarray = None):
        """
        Update all UAVs in swarm.

        Args:
            dt: Time step
            wind_force: Wind force vector (same for all UAVs or per-UAV)
        """
        # Update each UAV
        for uav in self.uavs:
            uav.update(dt, wind_force)

        # Check for collisions
        self._check_collisions()

        # Update swarm metrics
        self._update_swarm_metrics()

    def _check_collisions(self):
        """Check and handle collisions between UAVs."""
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(
                    self.uavs[i].state.position - self.uavs[j].state.position
                )
                collision_threshold = (self.uavs[i].collision_radius +
                                      self.uavs[j].collision_radius)

                if distance < collision_threshold:
                    self.uavs[i].collision_count += 1
                    self.uavs[j].collision_count += 1

                    # Simple collision response: push apart
                    if distance > 0:
                        push_direction = (self.uavs[i].state.position -
                                        self.uavs[j].state.position) / distance
                        push_distance = (collision_threshold - distance) / 2

                        self.uavs[i].state.position += push_direction * push_distance
                        self.uavs[j].state.position -= push_direction * push_distance

    def _update_swarm_metrics(self):
        """Update swarm-level metrics."""
        # Formation error (average deviation from desired formation)
        if len(self.uavs) > 1:
            positions = self.get_positions()
            centroid = np.mean(positions, axis=0)
            distances = np.linalg.norm(positions - centroid, axis=1)
            self.formation_error = np.std(distances)

        # Connectivity (percentage of UAVs that can communicate)
        connected_count = 0
        total_pairs = self.num_uavs * (self.num_uavs - 1) / 2

        if total_pairs > 0:
            for i in range(self.num_uavs):
                for j in range(i + 1, self.num_uavs):
                    if self.uavs[i].can_communicate_with(self.uavs[j]):
                        connected_count += 1
            self.connectivity = connected_count / total_pairs
        else:
            self.connectivity = 1.0

        # Coverage area (convex hull of UAV positions)
        if self.num_uavs >= 3:
            positions_2d = self.get_positions()[:, :2]
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(positions_2d)
                self.coverage_area = hull.volume
            except:
                self.coverage_area = 0.0

    def get_positions(self) -> np.ndarray:
        """Get all UAV positions."""
        return np.array([uav.get_position() for uav in self.uavs])

    def get_velocities(self) -> np.ndarray:
        """Get all UAV velocities."""
        return np.array([uav.get_velocity() for uav in self.uavs])

    def get_states(self) -> List[Dict]:
        """Get all UAV states."""
        return [uav.get_state_dict() for uav in self.uavs]

    def are_all_at_targets(self, threshold: float = 1.0) -> bool:
        """Check if all UAVs have reached their targets."""
        return all(uav.is_near_target(threshold) for uav in self.uavs)

    def get_total_distance(self) -> float:
        """Get total distance traveled by all UAVs."""
        return sum(uav.total_distance for uav in self.uavs)

    def get_total_energy(self) -> float:
        """Get total energy consumed by all UAVs."""
        return sum(uav.energy_consumed for uav in self.uavs)

    def get_min_battery_level(self) -> float:
        """Get minimum battery level across all UAVs."""
        return min(uav.state.battery_level for uav in self.uavs)

    def get_total_collisions(self) -> int:
        """Get total collision count."""
        return sum(uav.collision_count for uav in self.uavs) // 2  # Divide by 2 to avoid double counting

    def get_communication_graph(self) -> np.ndarray:
        """Get adjacency matrix of communication network."""
        graph = np.zeros((self.num_uavs, self.num_uavs))
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                if self.uavs[i].can_communicate_with(self.uavs[j]):
                    graph[i, j] = 1
                    graph[j, i] = 1
        return graph

    def reset(self, positions: np.ndarray = None):
        """Reset all UAVs."""
        if positions is not None:
            for i, uav in enumerate(self.uavs):
                uav.reset(positions[i])
        else:
            for uav in self.uavs:
                uav.reset()

        self.formation_error = 0.0
        self.connectivity = 1.0
        self.coverage_area = 0.0
