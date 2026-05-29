"""Benchmark runner for comparing algorithms across scenarios."""

import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
import time
import json
from pathlib import Path
import re
from .metrics import MetricsCalculator
from .multi_objective import calculate_mo_metrics


def normalize_algorithm_name(class_name: str) -> str:
    """
    Convert algorithm class name to config key format.
    Example: HybridPSOGWO -> hybrid_pso_gwo
    """
    # Insert underscores before uppercase letters (except first)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name)
    return name.lower()


def is_multi_objective_algorithm(algorithm_name: str) -> bool:
    """Check if algorithm is multi-objective."""
    mo_algorithms = ['mopso', 'mogwo', 'moaco']
    return algorithm_name.lower() in mo_algorithms


class BenchmarkRunner:
    """Run comprehensive benchmarks across algorithms and scenarios."""

    def __init__(self, config: Dict, results_dir: str = "results"):
        """Initialize benchmark runner."""
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_results = []
        self.metrics_calculator = MetricsCalculator(config)

    def run_benchmark(self,
                     algorithms: List[Any],
                     scenarios: List[Any],
                     num_trials: int = 30) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            algorithms: List of algorithm instances
            scenarios: List of scenario instances
            num_trials: Number of trials per combination

        Returns:
            Comprehensive benchmark results
        """
        print(f"\nRunning Benchmark: {len(algorithms)} algorithms x {len(scenarios)} scenarios x {num_trials} trials")
        print("=" * 80)

        all_results = []

        total_runs = len(algorithms) * len(scenarios) * num_trials
        pbar = tqdm(total=total_runs, desc="Benchmark Progress")

        for algorithm in algorithms:
            for scenario in scenarios:
                scenario_results = []

                for trial in range(num_trials):
                    # Reset scenario
                    scenario.reset()

                    # Run optimization
                    start_time = time.time()

                    # Get algorithm config using normalized name
                    algo_config_key = normalize_algorithm_name(algorithm.__class__.__name__)
                    max_iter = self.config.get('algorithms', {}).get(
                        algo_config_key, {}
                    ).get('max_iterations', 100)

                    # Check if algorithm is multi-objective
                    is_mo = is_multi_objective_algorithm(algorithm.__class__.__name__)

                    if is_mo:
                        # Multi-objective optimization
                        pareto_positions, pareto_objectives, stats = algorithm.optimize_multi_objective(
                            objective_function=scenario.evaluate_objectives,
                            bounds=scenario.get_bounds(),
                            max_iterations=max_iter
                        )

                        # Get best solution from Pareto front (min sum for comparison)
                        objective_sums = [np.sum(obj) for obj in pareto_objectives]
                        best_idx = np.argmin(objective_sums)
                        best_pos = pareto_positions[best_idx]
                        best_fit = objective_sums[best_idx]
                    else:
                        # Single-objective optimization
                        best_pos, best_fit, stats = algorithm.optimize(
                            objective_function=scenario.evaluate_solution,
                            bounds=scenario.get_bounds(),
                            max_iterations=max_iter
                        )
                        pareto_positions = None
                        pareto_objectives = None

                    run_time = time.time() - start_time

                    # Calculate comprehensive metrics
                    swarm_data = {
                        'num_uavs': algorithm.num_uavs,
                        'total_distance': 0,
                        'mission_time': run_time,
                        'coverage_ratio': 0,
                        'collision_count': 0,
                        'min_battery_level': 1.0,
                        'avg_battery_level': 1.0,
                        'avg_connectivity': 1.0,
                        'min_connectivity': 1.0,
                        'targets_reached': algorithm.num_uavs,
                    }

                    environment_data = {
                        'num_obstacles': 0,
                        'workspace_volume': 1000000,
                    }

                    stats['best_fitness'] = best_fit
                    stats['total_time'] = run_time

                    metrics = self.metrics_calculator.calculate_all_metrics(
                        algorithm.__class__.__name__,
                        scenario.get_name(),
                        stats,
                        swarm_data,
                        environment_data
                    )

                    # Calculate MO-specific metrics if applicable
                    if is_mo and pareto_objectives is not None:
                        # Calculate reference point (nadir point)
                        objectives_array = np.array(pareto_objectives)
                        reference_point = np.max(objectives_array, axis=0) * 1.1  # 10% beyond worst

                        mo_metrics = calculate_mo_metrics(
                            pareto_objectives,
                            reference_point=reference_point
                        )
                        metrics.update(mo_metrics)

                    # Collect results
                    result = {
                        'algorithm': algorithm.__class__.__name__,
                        'scenario': scenario.get_name(),
                        'trial': trial,
                        'best_fitness': best_fit,
                        'best_position': best_pos.tolist() if isinstance(best_pos, np.ndarray) else best_pos,
                        'run_time': run_time,
                        'statistics': stats,
                        'metrics': metrics,
                        'is_multi_objective': is_mo,
                    }

                    # Add MO-specific data
                    if is_mo and pareto_objectives is not None:
                        result['pareto_positions'] = [pos.tolist() if isinstance(pos, np.ndarray) else pos
                                                      for pos in pareto_positions]
                        result['pareto_objectives'] = [obj.tolist() if isinstance(obj, np.ndarray) else obj
                                                       for obj in pareto_objectives]

                    scenario_results.append(result)
                    all_results.append(result)
                    pbar.update(1)

                # Calculate aggregate statistics for this algorithm-scenario combination
                self._calculate_aggregate_stats(algorithm.__class__.__name__,
                                               scenario.get_name(),
                                               scenario_results)

        pbar.close()

        # Save results
        self._save_results(all_results)

        print("\nBenchmark Complete!")
        return {
            'results': all_results,
            'summary': self._generate_summary(all_results)
        }

    def _calculate_aggregate_stats(self,
                                   algorithm_name: str,
                                   scenario_name: str,
                                   results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate statistics across trials."""
        if not results:
            return {}

        fitness_values = [r['best_fitness'] for r in results]
        run_times = [r['run_time'] for r in results]

        stats = {
            'algorithm': algorithm_name,
            'scenario': scenario_name,
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'median_fitness': np.median(fitness_values),
            'best_fitness': np.min(fitness_values),
            'worst_fitness': np.max(fitness_values),
            'mean_time': np.mean(run_times),
            'std_time': np.std(run_times),
        }

        self.benchmark_results.append(stats)
        return stats

    def _generate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        summary = {
            'total_runs': len(results),
            'algorithms': list(set(r['algorithm'] for r in results)),
            'scenarios': list(set(r['scenario'] for r in results)),
            'by_algorithm': {},
            'by_scenario': {},
        }

        # Aggregate by algorithm
        for algo in summary['algorithms']:
            algo_results = [r for r in results if r['algorithm'] == algo]
            fitness_values = [r['best_fitness'] for r in algo_results]

            summary['by_algorithm'][algo] = {
                'mean_fitness': float(np.mean(fitness_values)),
                'std_fitness': float(np.std(fitness_values)),
                'best_fitness': float(np.min(fitness_values)),
            }

        # Aggregate by scenario
        for scenario in summary['scenarios']:
            scenario_results = [r for r in results if r['scenario'] == scenario]
            fitness_values = [r['best_fitness'] for r in scenario_results]

            summary['by_scenario'][scenario] = {
                'mean_fitness': float(np.mean(fitness_values)),
                'std_fitness': float(np.std(fitness_values)),
                'best_fitness': float(np.min(fitness_values)),
            }

        return summary

    def _save_results(self, results: List[Dict]):
        """Save benchmark results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.results_dir / f"benchmark_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save aggregate statistics
        if self.benchmark_results:
            stats_file = self.results_dir / f"benchmark_stats_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(self.benchmark_results, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print(f"Statistics saved to: {stats_file}")
