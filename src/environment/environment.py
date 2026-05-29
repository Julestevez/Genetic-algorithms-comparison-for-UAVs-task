"""Complete 3D environment for UAV simulation."""

import numpy as np
from typing import Dict, List, Tuple
from .obstacles import ObstacleManager


class Environment:
    """Complete simulation environment."""

    def __init__(self, config: Dict):
        """
        Initialize environment.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.env_config = config.get('environment', {})
        self.bounds = self.env_config.get('bounds', {})

        # Components
        self.obstacle_manager = ObstacleManager(config)

        # Coverage grid for area coverage scenarios
        self.grid_resolution = 5.0
        self.coverage_grid = None
        self._initialize_coverage_grid()

        # Targets for tracking scenarios
        self.targets = []
        self.target_velocities = []

        # Time
        self.time = 0.0

    def _initialize_coverage_grid(self):
        """Initialize grid for coverage tracking."""
        x_min = self.bounds.get('x_min', 0)
        x_max = self.bounds.get('x_max', 200)
        y_min = self.bounds.get('y_min', 0)
        y_max = self.bounds.get('y_max', 200)

        x_cells = int((x_max - x_min) / self.grid_resolution)
        y_cells = int((y_max - y_min) / self.grid_resolution)

        self.coverage_grid = np.zeros((x_cells, y_cells))
        self.grid_x_min = x_min
        self.grid_y_min = y_min

    def update(self, dt: float):
        """
        Update environment state.

        Args:
            dt: Time step
        """
        self.time += dt

        # Update dynamic obstacles
        self.obstacle_manager.update(dt)

        # Update targets
        for i in range(len(self.targets)):
            if i < len(self.target_velocities):
                self.targets[i] += self.target_velocities[i] * dt

                # Keep targets in bounds
                self.targets[i] = self._clamp_to_bounds(self.targets[i])

                # Random direction change
                if np.random.rand() < 0.01:
                    self.target_velocities[i] = np.random.randn(3)
                    speed = 5.0
                    vel_mag = np.linalg.norm(self.target_velocities[i])
                    if vel_mag > 0:
                        self.target_velocities[i] = (
                            self.target_velocities[i] / vel_mag * speed
                        )

    def _clamp_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """Clamp position to environment bounds."""
        x_min = self.bounds.get('x_min', 0)
        x_max = self.bounds.get('x_max', 200)
        y_min = self.bounds.get('y_min', 0)
        y_max = self.bounds.get('y_max', 200)
        z_min = self.bounds.get('z_min', 0)
        z_max = self.bounds.get('z_max', 100)

        clamped = position.copy()
        clamped[0] = np.clip(clamped[0], x_min, x_max)
        clamped[1] = np.clip(clamped[1], y_min, y_max)
        clamped[2] = np.clip(clamped[2], z_min, z_max)
        return clamped

    def update_coverage(self, uav_positions: np.ndarray, sensor_range: float):
        """
        Update coverage grid based on UAV positions.

        Args:
            uav_positions: Array of UAV positions
            sensor_range: Sensor range of UAVs
        """
        for pos in uav_positions:
            # Convert position to grid coordinates
            x_idx = int((pos[0] - self.grid_x_min) / self.grid_resolution)
            y_idx = int((pos[1] - self.grid_y_min) / self.grid_resolution)

            # Mark nearby cells as covered
            radius_cells = int(sensor_range / self.grid_resolution)

            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    nx, ny = x_idx + dx, y_idx + dy

                    if (0 <= nx < self.coverage_grid.shape[0] and
                        0 <= ny < self.coverage_grid.shape[1]):

                        cell_pos = np.array([
                            self.grid_x_min + nx * self.grid_resolution,
                            self.grid_y_min + ny * self.grid_resolution,
                            pos[2]
                        ])

                        if np.linalg.norm(cell_pos[:2] - pos[:2]) <= sensor_range:
                            self.coverage_grid[nx, ny] += 1

    def get_coverage_ratio(self) -> float:
        """Get ratio of covered area."""
        if self.coverage_grid is None:
            return 0.0
        total_cells = self.coverage_grid.size
        covered_cells = np.sum(self.coverage_grid > 0)
        return covered_cells / total_cells if total_cells > 0 else 0.0

    def add_target(self, position: np.ndarray, velocity: np.ndarray = None):
        """Add a target to track."""
        self.targets.append(position.copy())
        if velocity is not None:
            self.target_velocities.append(velocity.copy())
        else:
            self.target_velocities.append(np.zeros(3))

    def get_targets(self) -> List[np.ndarray]:
        """Get all target positions."""
        return [t.copy() for t in self.targets]

    def reset(self):
        """Reset environment."""
        self.time = 0.0
        self.obstacle_manager.clear()
        self._initialize_coverage_grid()
        self.targets = []
        self.target_velocities = []
