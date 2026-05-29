"""Physics engine for UAV simulation including wind and environmental effects."""

import numpy as np
from typing import Dict, Tuple


class PhysicsEngine:
    """Physics engine for environmental effects."""

    def __init__(self, config: Dict):
        """
        Initialize physics engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config.get('environment', {})
        self.wind_config = self.config.get('wind', {})

        self.wind_enabled = self.wind_config.get('enabled', True)
        self.max_wind_speed = self.wind_config.get('max_speed', 5.0)
        self.turbulence = self.wind_config.get('turbulence', 0.5)

        # Wind state
        self.base_wind = np.zeros(3)
        self.wind_time = 0.0

        # Constants
        self.gravity = 9.81
        self.air_density = 1.225  # kg/m^3

    def update(self, dt: float):
        """
        Update physics state.

        Args:
            dt: Time step
        """
        self.wind_time += dt

        if self.wind_enabled:
            # Generate base wind with smooth variation
            wind_frequency = 0.1
            self.base_wind[0] = self.max_wind_speed * np.sin(
                2 * np.pi * wind_frequency * self.wind_time
            )
            self.base_wind[1] = self.max_wind_speed * np.cos(
                2 * np.pi * wind_frequency * self.wind_time * 0.7
            )
            self.base_wind[2] = self.max_wind_speed * 0.3 * np.sin(
                2 * np.pi * wind_frequency * self.wind_time * 1.3
            )

    def get_wind_force(self, position: np.ndarray) -> np.ndarray:
        """
        Get wind force at a given position.

        Args:
            position: 3D position

        Returns:
            Wind force vector
        """
        if not self.wind_enabled:
            return np.zeros(3)

        # Add turbulence based on position
        turbulence_component = np.random.randn(3) * self.turbulence * self.max_wind_speed

        # Add height-dependent wind (stronger at higher altitudes)
        height_factor = 1.0 + position[2] / 100.0

        wind = self.base_wind * height_factor + turbulence_component

        # Convert to force (simplified drag model)
        # F = 0.5 * rho * Cd * A * v^2
        drag_coefficient = 0.5
        reference_area = 0.1  # m^2
        wind_magnitude = np.linalg.norm(wind)

        if wind_magnitude > 0:
            force_magnitude = (0.5 * self.air_density * drag_coefficient *
                             reference_area * wind_magnitude ** 2)
            wind_force = (wind / wind_magnitude) * force_magnitude
        else:
            wind_force = np.zeros(3)

        return wind_force

    def get_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate air drag force.

        Args:
            velocity: UAV velocity

        Returns:
            Drag force vector
        """
        speed = np.linalg.norm(velocity)
        if speed < 0.01:
            return np.zeros(3)

        drag_coefficient = 0.4
        reference_area = 0.05  # m^2

        drag_magnitude = (0.5 * self.air_density * drag_coefficient *
                         reference_area * speed ** 2)

        drag_direction = -velocity / speed
        return drag_direction * drag_magnitude

    def check_collision_with_ground(self, position: np.ndarray) -> bool:
        """
        Check if position is below ground.

        Args:
            position: 3D position

        Returns:
            True if collision detected
        """
        return position[2] < 0

    def check_out_of_bounds(self, position: np.ndarray) -> bool:
        """
        Check if position is out of environment bounds.

        Args:
            position: 3D position

        Returns:
            True if out of bounds
        """
        bounds = self.config.get('bounds', {})
        x_min = bounds.get('x_min', 0)
        x_max = bounds.get('x_max', 200)
        y_min = bounds.get('y_min', 0)
        y_max = bounds.get('y_max', 200)
        z_min = bounds.get('z_min', 0)
        z_max = bounds.get('z_max', 100)

        return (position[0] < x_min or position[0] > x_max or
                position[1] < y_min or position[1] > y_max or
                position[2] < z_min or position[2] > z_max)

    def apply_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        Apply boundary constraints to position.

        Args:
            position: 3D position

        Returns:
            Constrained position
        """
        bounds = self.config.get('bounds', {})
        x_min = bounds.get('x_min', 0)
        x_max = bounds.get('x_max', 200)
        y_min = bounds.get('y_min', 0)
        y_max = bounds.get('y_max', 200)
        z_min = bounds.get('z_min', 0)
        z_max = bounds.get('z_max', 100)

        constrained = position.copy()
        constrained[0] = np.clip(constrained[0], x_min, x_max)
        constrained[1] = np.clip(constrained[1], y_min, y_max)
        constrained[2] = np.clip(constrained[2], z_min, z_max)

        return constrained

    def reset(self):
        """Reset physics state."""
        self.base_wind = np.zeros(3)
        self.wind_time = 0.0
