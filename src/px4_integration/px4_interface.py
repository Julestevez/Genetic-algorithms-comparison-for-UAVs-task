"""Interface for PX4 SITL integration."""

import asyncio
import numpy as np
from typing import Dict, List
from mavsdk import System


class PX4Interface:
    """Interface to PX4 SITL for hardware-in-the-loop simulation."""

    def __init__(self, config: Dict, num_uavs: int):
        """
        Initialize PX4 interface.

        Args:
            config: Configuration dictionary
            num_uavs: Number of UAVs
        """
        self.config = config.get('px4', {})
        self.num_uavs = num_uavs
        self.enabled = self.config.get('enabled', False)

        if not self.enabled:
            return

        self.base_port = self.config.get('mavlink_port', 14540)
        self.systems = []

        self.home_position = self.config.get('home_position', [47.397742, 8.545594, 488.0])

    async def connect(self):
        """Connect to PX4 SITL instances."""
        if not self.enabled:
            return

        print(f"Connecting to {self.num_uavs} PX4 SITL instances...")

        for i in range(self.num_uavs):
            system = System()
            port = self.base_port + i
            await system.connect(f"udp://:{port}")

            print(f"Waiting for UAV {i} to connect...")
            async for state in system.core.connection_state():
                if state.is_connected:
                    print(f"UAV {i} connected!")
                    break

            self.systems.append(system)

        print("All UAVs connected!")

    async def arm_and_takeoff(self, altitude: float = 10.0):
        """Arm all UAVs and takeoff."""
        if not self.enabled:
            return

        tasks = []
        for i, system in enumerate(self.systems):
            tasks.append(self._arm_and_takeoff_single(system, i, altitude))

        await asyncio.gather(*tasks)

    async def _arm_and_takeoff_single(self, system: System, uav_id: int, altitude: float):
        """Arm and takeoff single UAV."""
        print(f"Arming UAV {uav_id}...")
        await system.action.arm()

        print(f"Taking off UAV {uav_id}...")
        await system.action.set_takeoff_altitude(altitude)
        await system.action.takeoff()

        # Wait for takeoff to complete
        await asyncio.sleep(5)

    async def set_positions(self, positions: np.ndarray):
        """
        Set target positions for all UAVs.

        Args:
            positions: Array of shape (num_uavs, 3) - NED coordinates
        """
        if not self.enabled:
            return

        tasks = []
        for i, system in enumerate(self.systems):
            if i < len(positions):
                ned_pos = positions[i]  # Assuming already in NED
                tasks.append(self._set_position_single(system, ned_pos))

        await asyncio.gather(*tasks)

    async def _set_position_single(self, system: System, ned_position: np.ndarray):
        """Set position for single UAV."""
        await system.offboard.set_position_ned(
            ned_position[0],
            ned_position[1],
            ned_position[2],
            0.0  # yaw
        )

    async def land_all(self):
        """Land all UAVs."""
        if not self.enabled:
            return

        print("Landing all UAVs...")
        tasks = []
        for system in self.systems:
            tasks.append(system.action.land())

        await asyncio.gather(*tasks)

    async def disconnect(self):
        """Disconnect from all PX4 instances."""
        if not self.enabled:
            return

        print("Disconnecting from PX4 SITL...")
        # Cleanup would go here

    def convert_to_ned(self, position: np.ndarray) -> np.ndarray:
        """
        Convert from simulation coordinates to NED.

        Args:
            position: Position in simulation frame (x, y, z)

        Returns:
            Position in NED frame (north, east, down)
        """
        # Simple conversion - in real implementation would use proper coordinate transform
        ned = np.array([
            position[0],  # North
            position[1],  # East
            -position[2]  # Down (negative z)
        ])
        return ned

    def is_enabled(self) -> bool:
        """Check if PX4 integration is enabled."""
        return self.enabled
