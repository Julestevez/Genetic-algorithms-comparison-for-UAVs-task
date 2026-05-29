"""Obstacle management for 3D environment."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Obstacle:
    """3D obstacle representation."""
    position: np.ndarray  # Center position
    size: np.ndarray      # Size in x, y, z
    velocity: np.ndarray = None  # For dynamic obstacles
    obstacle_type: str = "static"  # static or dynamic

    def contains_point(self, point: np.ndarray, safety_margin: float = 0.0) -> bool:
        """Check if point is inside obstacle (with optional safety margin)."""
        half_size = self.size / 2 + safety_margin
        relative_pos = np.abs(point - self.position)
        return np.all(relative_pos <= half_size)

    def distance_to_point(self, point: np.ndarray) -> float:
        """Calculate minimum distance from point to obstacle surface."""
        # Calculate closest point on obstacle surface
        half_size = self.size / 2
        closest = np.clip(
            point - self.position,
            -half_size,
            half_size
        ) + self.position

        return np.linalg.norm(point - closest)

    def update(self, dt: float, bounds: Tuple[float, float, float, float, float, float]):
        """Update dynamic obstacle position."""
        if self.velocity is not None:
            self.position += self.velocity * dt

            # Bounce off boundaries
            x_min, x_max, y_min, y_max, z_min, z_max = bounds

            if self.position[0] <= x_min or self.position[0] >= x_max:
                self.velocity[0] *= -1
                self.position[0] = np.clip(self.position[0], x_min, x_max)

            if self.position[1] <= y_min or self.position[1] >= y_max:
                self.velocity[1] *= -1
                self.position[1] = np.clip(self.position[1], y_min, y_max)

            if self.position[2] <= z_min or self.position[2] >= z_max:
                self.velocity[2] *= -1
                self.position[2] = np.clip(self.position[2], z_min, z_max)


class ObstacleManager:
    """Manages static and dynamic obstacles."""

    def __init__(self, config: Dict):
        """
        Initialize obstacle manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config.get('environment', {})
        self.obstacle_config = self.config.get('obstacles', {})
        self.bounds = self._get_bounds()

        self.obstacles: List[Obstacle] = []

    def _get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """Get environment bounds."""
        bounds = self.config.get('bounds', {})
        return (
            bounds.get('x_min', 0),
            bounds.get('x_max', 200),
            bounds.get('y_min', 0),
            bounds.get('y_max', 200),
            bounds.get('z_min', 0),
            bounds.get('z_max', 100)
        )

    def generate_random_obstacles(self, num_static: int = None, num_dynamic: int = None):
        """Generate random obstacles in the environment."""
        if num_static is None:
            num_static = self.obstacle_config.get('num_static', 20)
        if num_dynamic is None:
            num_dynamic = self.obstacle_config.get('num_dynamic', 5)

        min_size = self.obstacle_config.get('min_size', 2.0)
        max_size = self.obstacle_config.get('max_size', 10.0)
        dynamic_velocity = self.obstacle_config.get('dynamic_velocity', 2.0)

        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds

        # Generate static obstacles
        for _ in range(num_static):
            position = np.array([
                np.random.uniform(x_min + 10, x_max - 10),
                np.random.uniform(y_min + 10, y_max - 10),
                np.random.uniform(z_min + 5, z_max - 5)
            ])

            size = np.random.uniform(min_size, max_size, 3)

            self.obstacles.append(Obstacle(
                position=position,
                size=size,
                obstacle_type="static"
            ))

        # Generate dynamic obstacles
        for _ in range(num_dynamic):
            position = np.array([
                np.random.uniform(x_min + 10, x_max - 10),
                np.random.uniform(y_min + 10, y_max - 10),
                np.random.uniform(z_min + 5, z_max - 5)
            ])

            size = np.random.uniform(min_size, max_size, 3)

            velocity = np.random.randn(3)
            velocity = velocity / np.linalg.norm(velocity) * dynamic_velocity

            self.obstacles.append(Obstacle(
                position=position,
                size=size,
                velocity=velocity,
                obstacle_type="dynamic"
            ))

    def generate_structured_obstacles(self, pattern: str = "corridor"):
        """
        Generate obstacles in specific patterns.

        Args:
            pattern: Pattern type (corridor, forest, maze, etc.)
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds

        if pattern == "corridor":
            # Create corridor with obstacles on sides
            for i in range(10):
                z = z_min + 10 + i * 8
                # Left wall obstacles
                self.obstacles.append(Obstacle(
                    position=np.array([x_min + 20, (y_min + y_max) / 2, z]),
                    size=np.array([5, 10, 5]),
                    obstacle_type="static"
                ))
                # Right wall obstacles
                self.obstacles.append(Obstacle(
                    position=np.array([x_max - 20, (y_min + y_max) / 2, z]),
                    size=np.array([5, 10, 5]),
                    obstacle_type="static"
                ))

        elif pattern == "forest":
            # Random "trees" (tall vertical obstacles)
            for _ in range(30):
                position = np.array([
                    np.random.uniform(x_min + 10, x_max - 10),
                    np.random.uniform(y_min + 10, y_max - 10),
                    (z_max - z_min) / 2
                ])
                self.obstacles.append(Obstacle(
                    position=position,
                    size=np.array([3, 3, z_max - z_min]),
                    obstacle_type="static"
                ))

        elif pattern == "maze":
            # Grid-based maze
            grid_size = 20
            for i in range(5):
                for j in range(5):
                    if np.random.rand() > 0.5:
                        position = np.array([
                            x_min + 20 + i * grid_size,
                            y_min + 20 + j * grid_size,
                            (z_max - z_min) / 2
                        ])
                        self.obstacles.append(Obstacle(
                            position=position,
                            size=np.array([grid_size * 0.8, grid_size * 0.8, 30]),
                            obstacle_type="static"
                        ))

    def update(self, dt: float):
        """Update dynamic obstacles."""
        for obstacle in self.obstacles:
            if obstacle.velocity is not None:
                obstacle.update(dt, self.bounds)

    def is_position_valid(self, position: np.ndarray, safety_margin: float = 1.0) -> bool:
        """Check if position is collision-free."""
        for obstacle in self.obstacles:
            if obstacle.contains_point(position, safety_margin):
                return False
        return True

    def get_closest_obstacle(self, position: np.ndarray) -> Tuple[Obstacle, float]:
        """Get closest obstacle and distance to it."""
        if not self.obstacles:
            return None, float('inf')

        min_distance = float('inf')
        closest_obstacle = None

        for obstacle in self.obstacles:
            distance = obstacle.distance_to_point(position)
            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle

        return closest_obstacle, min_distance

    def get_nearby_obstacles(self, position: np.ndarray, radius: float) -> List[Obstacle]:
        """Get all obstacles within radius of position."""
        nearby = []
        for obstacle in self.obstacles:
            distance = obstacle.distance_to_point(position)
            if distance <= radius:
                nearby.append(obstacle)
        return nearby

    def clear(self):
        """Remove all obstacles."""
        self.obstacles = []

    def get_obstacle_data(self) -> List[Dict]:
        """Get obstacle data for visualization."""
        return [
            {
                'position': obs.position.copy(),
                'size': obs.size.copy(),
                'type': obs.obstacle_type,
                'velocity': obs.velocity.copy() if obs.velocity is not None else None
            }
            for obs in self.obstacles
        ]
