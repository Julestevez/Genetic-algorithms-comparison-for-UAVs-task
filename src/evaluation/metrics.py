"""Comprehensive metrics calculation for UAV swarm performance."""

import numpy as np
from typing import Dict, List, Any
from scipy import stats
import networkx as nx


class MetricsCalculator:
    """Calculate 30+ evaluation metrics for UAV swarm performance."""

    def __init__(self, config: Dict):
        """Initialize metrics calculator."""
        self.config = config
        self.metrics = {}

    def calculate_all_metrics(self,
                             algorithm_name: str,
                             scenario_name: str,
                             optimization_stats: Dict,
                             swarm_data: Dict,
                             environment_data: Dict) -> Dict[str, Any]:
        """
        Calculate all metrics for a simulation run.

        Args:
            algorithm_name: Name of algorithm used
            scenario_name: Name of scenario
            optimization_stats: Statistics from optimization
            swarm_data: Data from swarm simulation
            environment_data: Environment information

        Returns:
            Dictionary of all calculated metrics
        """
        metrics = {
            'algorithm': algorithm_name,
            'scenario': scenario_name,
        }

        # Optimization metrics
        metrics.update(self._calculate_optimization_metrics(optimization_stats))

        # Performance metrics
        metrics.update(self._calculate_performance_metrics(swarm_data))

        # Safety metrics
        metrics.update(self._calculate_safety_metrics(swarm_data, environment_data))

        # Efficiency metrics
        metrics.update(self._calculate_efficiency_metrics(swarm_data))

        # Coordination metrics
        metrics.update(self._calculate_coordination_metrics(swarm_data))

        # Convergence metrics
        metrics.update(self._calculate_convergence_metrics(optimization_stats))

        # Scalability metrics
        metrics.update(self._calculate_scalability_metrics(swarm_data))

        return metrics

    def _calculate_optimization_metrics(self, stats: Dict) -> Dict[str, float]:
        """Calculate metrics related to optimization process."""
        return {
            'best_fitness': stats.get('best_fitness', float('inf')),
            'num_evaluations': stats.get('num_evaluations', 0),
            'total_time': stats.get('total_time', 0),
            'convergence_iteration': stats.get('convergence_iteration', 0),
            'final_diversity': stats.get('final_diversity', 0),
            'avg_iteration_time': stats.get('avg_iteration_time', 0),
        }

    def _calculate_performance_metrics(self, swarm_data: Dict) -> Dict[str, float]:
        """Calculate performance-related metrics."""
        metrics = {}

        # Success rate
        if 'targets_reached' in swarm_data:
            metrics['success_rate'] = swarm_data['targets_reached'] / swarm_data.get('num_uavs', 1)
        else:
            metrics['success_rate'] = 0.0

        # Completion time
        metrics['completion_time'] = swarm_data.get('mission_time', 0)

        # Path metrics
        metrics['total_path_length'] = swarm_data.get('total_distance', 0)
        metrics['avg_path_length'] = (swarm_data.get('total_distance', 0) /
                                     swarm_data.get('num_uavs', 1))

        # Coverage metrics
        metrics['coverage_ratio'] = swarm_data.get('coverage_ratio', 0)
        metrics['coverage_efficiency'] = (swarm_data.get('coverage_ratio', 0) /
                                         (swarm_data.get('total_distance', 1) + 1e-10))

        # Formation metrics
        metrics['avg_formation_error'] = swarm_data.get('avg_formation_error', 0)
        metrics['max_formation_error'] = swarm_data.get('max_formation_error', 0)

        return metrics

    def _calculate_safety_metrics(self, swarm_data: Dict, environment_data: Dict) -> Dict[str, float]:
        """Calculate safety-related metrics."""
        metrics = {}

        # Collision metrics
        metrics['collision_count'] = swarm_data.get('collision_count', 0)
        metrics['collision_rate'] = (swarm_data.get('collision_count', 0) /
                                    max(swarm_data.get('mission_time', 1), 1))

        # Near-miss events (close calls)
        metrics['near_miss_count'] = swarm_data.get('near_miss_count', 0)

        # Obstacle proximity
        metrics['min_obstacle_distance'] = swarm_data.get('min_obstacle_distance', float('inf'))
        metrics['avg_obstacle_distance'] = swarm_data.get('avg_obstacle_distance', 0)

        # Safety violations
        metrics['safety_violations'] = swarm_data.get('safety_violations', 0)

        # Minimum inter-UAV separation
        metrics['min_separation'] = swarm_data.get('min_separation', 0)
        metrics['avg_separation'] = swarm_data.get('avg_separation', 0)

        return metrics

    def _calculate_efficiency_metrics(self, swarm_data: Dict) -> Dict[str, float]:
        """Calculate efficiency-related metrics."""
        metrics = {}

        # Energy metrics
        metrics['total_energy_consumed'] = swarm_data.get('total_energy', 0)
        metrics['avg_energy_per_uav'] = (swarm_data.get('total_energy', 0) /
                                        swarm_data.get('num_uavs', 1))
        metrics['energy_efficiency'] = (swarm_data.get('total_distance', 0) /
                                       max(swarm_data.get('total_energy', 1), 1e-10))

        # Battery metrics
        metrics['min_battery_level'] = swarm_data.get('min_battery_level', 0)
        metrics['avg_battery_level'] = swarm_data.get('avg_battery_level', 0)

        # Time efficiency
        total_distance = swarm_data.get('total_distance', 0)
        completion_time = swarm_data.get('mission_time', 1)
        metrics['avg_velocity'] = total_distance / max(completion_time, 1)

        # Resource utilization
        metrics['uav_utilization'] = swarm_data.get('active_uavs', 0) / swarm_data.get('num_uavs', 1)

        return metrics

    def _calculate_coordination_metrics(self, swarm_data: Dict) -> Dict[str, float]:
        """Calculate coordination and communication metrics."""
        metrics = {}

        # Connectivity
        metrics['avg_connectivity'] = swarm_data.get('avg_connectivity', 0)
        metrics['min_connectivity'] = swarm_data.get('min_connectivity', 0)

        # Network properties
        if 'communication_graphs' in swarm_data:
            graphs = swarm_data['communication_graphs']
            if graphs:
                # Average over all timesteps
                clustering_coeffs = []
                path_lengths = []

                for adj_matrix in graphs:
                    G = nx.from_numpy_array(adj_matrix)
                    if G.number_of_nodes() > 0:
                        try:
                            clustering = nx.average_clustering(G)
                            clustering_coeffs.append(clustering)
                        except:
                            pass

                        if nx.is_connected(G):
                            try:
                                avg_path = nx.average_shortest_path_length(G)
                                path_lengths.append(avg_path)
                            except:
                                pass

                metrics['avg_clustering_coefficient'] = (np.mean(clustering_coeffs)
                                                        if clustering_coeffs else 0)
                metrics['avg_path_length_network'] = (np.mean(path_lengths)
                                                     if path_lengths else 0)

        # Synchronization
        metrics['synchronization_index'] = swarm_data.get('synchronization', 0)

        # Load balancing
        if 'uav_workloads' in swarm_data:
            workloads = swarm_data['uav_workloads']
            metrics['load_balance_std'] = np.std(workloads)
            metrics['load_balance_coefficient'] = (np.std(workloads) /
                                                  max(np.mean(workloads), 1e-10))

        return metrics

    def _calculate_convergence_metrics(self, stats: Dict) -> Dict[str, float]:
        """Calculate convergence-related metrics."""
        metrics = {}

        # Convergence speed
        metrics['convergence_speed'] = 1.0 / (stats.get('convergence_iteration', 1) + 1)

        # Final convergence
        if 'best_fitness_history' in stats:
            history = stats['best_fitness_history']
            if len(history) > 10:
                # Improvement rate in last 10 iterations
                recent_improvement = abs(history[-1] - history[-10])
                metrics['final_improvement_rate'] = recent_improvement / 10

                # Total improvement
                if abs(history[0]) > 1e-10:
                    metrics['total_improvement'] = abs(history[-1] - history[0]) / abs(history[0])
                else:
                    metrics['total_improvement'] = 0

                # Convergence stability (lower is better)
                metrics['convergence_stability'] = np.std(history[-10:]) if len(history) >= 10 else 0

        # Diversity maintenance
        if 'diversity_history' in stats:
            diversity = stats['diversity_history']
            if diversity:
                metrics['diversity_retention'] = diversity[-1] / (diversity[0] + 1e-10)

        return metrics

    def _calculate_scalability_metrics(self, swarm_data: Dict) -> Dict[str, float]:
        """Calculate scalability-related metrics."""
        metrics = {}

        num_uavs = swarm_data.get('num_uavs', 1)

        # Per-UAV performance
        metrics['performance_per_uav'] = (swarm_data.get('coverage_ratio', 0) /
                                         max(num_uavs, 1))

        # Scalability index (performance / computational cost)
        total_time = swarm_data.get('mission_time', 1)
        metrics['scalability_index'] = (swarm_data.get('coverage_ratio', 0) /
                                       (num_uavs * total_time + 1e-10))

        # Communication overhead
        if 'messages_sent' in swarm_data:
            metrics['messages_per_uav'] = (swarm_data['messages_sent'] /
                                          max(num_uavs, 1))
            metrics['communication_overhead'] = (swarm_data['messages_sent'] /
                                                max(total_time, 1))

        return metrics

    def calculate_robustness_score(self, metrics_across_trials: List[Dict]) -> float:
        """
        Calculate robustness score based on performance variance across trials.

        Args:
            metrics_across_trials: List of metric dictionaries from multiple trials

        Returns:
            Robustness score (0-1, higher is better)
        """
        if len(metrics_across_trials) < 2:
            return 1.0

        # Extract key performance metrics
        success_rates = [m.get('success_rate', 0) for m in metrics_across_trials]
        completion_times = [m.get('completion_time', 0) for m in metrics_across_trials]
        collision_counts = [m.get('collision_count', 0) for m in metrics_across_trials]

        # Calculate coefficient of variation for each metric
        cv_success = np.std(success_rates) / (np.mean(success_rates) + 1e-10)
        cv_time = np.std(completion_times) / (np.mean(completion_times) + 1e-10)
        cv_collision = np.std(collision_counts) / (np.mean(collision_counts) + 1)

        # Lower CV = higher robustness
        robustness = 1.0 / (1.0 + 0.33 * (cv_success + cv_time + cv_collision))

        return robustness
