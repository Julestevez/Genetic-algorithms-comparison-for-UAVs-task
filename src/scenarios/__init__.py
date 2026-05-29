"""Benchmark scenarios for UAV swarm optimization - Single-Objective Paper."""

from .base_scenario import BaseScenario
from .obstacle_avoidance import ObstacleAvoidanceScenario
from .area_coverage import AreaCoverageScenario
from .target_tracking import TargetTrackingScenario
from .formation_flight import FormationFlightScenario
from .multi_target_engagement import MultiTargetEngagementScenario
from .dynamic_obstacle_avoidance import DynamicObstacleAvoidanceScenario

__all__ = [
    'BaseScenario',
    'ObstacleAvoidanceScenario',
    'AreaCoverageScenario',
    'TargetTrackingScenario',
    'FormationFlightScenario',
    'MultiTargetEngagementScenario',
    'DynamicObstacleAvoidanceScenario',
]

# Scenario registry - 6 scenarios for Paper A
SCENARIOS = {
    'obstacle_avoidance': ObstacleAvoidanceScenario,
    'area_coverage': AreaCoverageScenario,
    'target_tracking': TargetTrackingScenario,
    'formation_flight': FormationFlightScenario,
    'multi_target_engagement': MultiTargetEngagementScenario,
    'dynamic_obstacle_avoidance': DynamicObstacleAvoidanceScenario,
}
