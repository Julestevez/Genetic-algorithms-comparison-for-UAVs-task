"""3D environment with obstacles and spatial structures."""

from .obstacles import Obstacle, ObstacleManager
from .environment import Environment

__all__ = ['Obstacle', 'ObstacleManager', 'Environment']
