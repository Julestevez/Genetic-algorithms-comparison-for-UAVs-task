"""Individual UAV with full dynamics and physics."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class UAVState:
    """Complete state of a UAV."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # roll, pitch, yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    battery_level: float = 1.0
    time: float = 0.0


class UAV:
    """Individual UAV with physics simulation."""

    def __init__(self, uav_id: int, config: Dict, initial_position: np.ndarray = None):
        """
        Initialize UAV.

        Args:
            uav_id: Unique identifier
            config: Configuration dictionary
            initial_position: Initial 3D position
        """
        self.id = uav_id
        self.config = config.get('uav', {})

        # Physical parameters
        self.mass = self.config.get('mass', 1.5)
        self.max_velocity = self.config.get('max_velocity', 15.0)
        self.max_acceleration = self.config.get('max_acceleration', 5.0)
        self.max_angular_velocity = self.config.get('max_angular_velocity', 2.0)
        self.collision_radius = self.config.get('collision_radius', 0.5)
        self.communication_range = self.config.get('communication_range', 100.0)
        self.sensor_range = self.config.get('sensor_range', 50.0)

        # Battery parameters
        self.battery_capacity = self.config.get('battery_capacity', 5000)
        self.power_consumption = self.config.get('power_consumption', 100)
        self.current_battery = self.battery_capacity

        # State
        if initial_position is not None:
            self.state = UAVState(position=initial_position.copy())
        else:
            self.state = UAVState()

        # Control
        self.target_position = None
        self.target_velocity = np.zeros(3)

        # History
        self.position_history = [self.state.position.copy()]
        self.velocity_history = [self.state.velocity.copy()]
        self.battery_history = [1.0]

        # Metrics
        self.total_distance = 0.0
        self.energy_consumed = 0.0
        self.collision_count = 0
        self.mission_time = 0.0

    def set_target(self, target_position: np.ndarray):
        """Set target position for UAV."""
        self.target_position = target_position.copy()

    def update(self, dt: float, wind_force: np.ndarray = None):
        """
        Update UAV state with physics simulation.

        Args:
            dt: Time step
            wind_force: External wind force vector
        """
        if self.target_position is None:
            return

        # Calculate desired velocity toward target
        direction = self.target_position - self.state.position
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            desired_velocity = (direction / distance) * self.max_velocity
            # Slow down when approaching target
            if distance < 5.0:
                desired_velocity *= (distance / 5.0)
        else:
            desired_velocity = np.zeros(3)

        # Calculate acceleration (PID-like control)
        velocity_error = desired_velocity - self.state.velocity
        self.state.acceleration = np.clip(
            velocity_error / dt,
            -self.max_acceleration,
            self.max_acceleration
        )

        # Apply wind force if present
        if wind_force is not None:
            wind_acceleration = wind_force / self.mass
            self.state.acceleration += wind_acceleration

        # Update velocity
        self.state.velocity += self.state.acceleration * dt
        speed = np.linalg.norm(self.state.velocity)
        if speed > self.max_velocity:
            self.state.velocity = (self.state.velocity / speed) * self.max_velocity

        # Update position
        old_position = self.state.position.copy()
        self.state.position += self.state.velocity * dt

        # Update orientation (point in direction of motion)
        if speed > 0.1:
            # Yaw from velocity direction
            self.state.orientation[2] = np.arctan2(
                self.state.velocity[1],
                self.state.velocity[0]
            )
            # Pitch from vertical velocity
            self.state.orientation[1] = np.arcsin(
                np.clip(self.state.velocity[2] / speed, -1, 1)
            )

        # Update battery
        power = self.power_consumption + (speed / self.max_velocity) * 50
        self.current_battery -= power * dt / 1000.0
        self.state.battery_level = self.current_battery / self.battery_capacity

        # Update metrics
        step_distance = np.linalg.norm(self.state.position - old_position)
        self.total_distance += step_distance
        self.energy_consumed += power * dt / 3600.0  # Convert to Wh
        self.mission_time += dt
        self.state.time += dt

        # Update history
        self.position_history.append(self.state.position.copy())
        self.velocity_history.append(self.state.velocity.copy())
        self.battery_history.append(self.state.battery_level)

    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.state.position.copy()

    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.state.velocity.copy()

    def is_near_target(self, threshold: float = 1.0) -> bool:
        """Check if UAV is near its target."""
        if self.target_position is None:
            return False
        distance = np.linalg.norm(self.target_position - self.state.position)
        return distance < threshold

    def get_state_dict(self) -> Dict:
        """Get complete state as dictionary."""
        return {
            'id': self.id,
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'acceleration': self.state.acceleration.copy(),
            'orientation': self.state.orientation.copy(),
            'battery_level': self.state.battery_level,
            'time': self.state.time,
            'total_distance': self.total_distance,
            'energy_consumed': self.energy_consumed,
        }

    def can_communicate_with(self, other: 'UAV') -> bool:
        """Check if this UAV can communicate with another."""
        distance = np.linalg.norm(self.state.position - other.state.position)
        return distance <= self.communication_range

    def can_sense(self, position: np.ndarray) -> bool:
        """Check if UAV can sense a position."""
        distance = np.linalg.norm(self.state.position - position)
        return distance <= self.sensor_range

    def reset(self, position: np.ndarray = None):
        """Reset UAV to initial state."""
        if position is not None:
            self.state.position = position.copy()
        self.state.velocity = np.zeros(3)
        self.state.acceleration = np.zeros(3)
        self.state.orientation = np.zeros(3)
        self.state.battery_level = 1.0
        self.current_battery = self.battery_capacity
        self.state.time = 0.0

        self.position_history = [self.state.position.copy()]
        self.velocity_history = [self.state.velocity.copy()]
        self.battery_history = [1.0]

        self.total_distance = 0.0
        self.energy_consumed = 0.0
        self.collision_count = 0
        self.mission_time = 0.0
        self.target_position = None
