"""Main application for UAV swarm optimization benchmark."""

import argparse
import sys
from pathlib import Path
import numpy as np
import os

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Change to project root directory
os.chdir(Path(__file__).parent.parent)

from src.algorithms import PSO, GWO, ACO, MOPSO, MOGWO, MOACO
from src.scenarios import SCENARIOS
from src.evaluation import MetricsCalculator, BenchmarkRunner, StatisticalAnalyzer
from src.visualization import Plotter, Animator
from src.utils import setup_logger, load_config


class UAVSwarmSimulation:
    """Main simulation controller."""

    def __init__(self, config_path: str = "configs/default_config.yaml"):
        """Initialize simulation."""
        self.config = load_config(config_path)
        self.logger = setup_logger(level=self.config.get('logging', {}).get('level', 'INFO'))

        self.logger.info("UAV Swarm Optimization Framework Initialized")
        self.logger.info(f"Configuration loaded from: {config_path}")

        # Initialize components
        self.metrics_calculator = MetricsCalculator(self.config)
        self.plotter = Plotter()
        self.animator = Animator()

    def get_algorithm(self, algorithm_name: str, num_uavs: int):
        """Get algorithm instance by name."""
        algorithms = {
            'pso': PSO,
            'gwo': GWO,
            'aco': ACO,
            'mopso': MOPSO,
            'mogwo': MOGWO,
            'moaco': MOACO,
        }

        if algorithm_name.lower() not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        return algorithms[algorithm_name.lower()](self.config, num_uavs)

    def get_scenario(self, scenario_name: str, num_uavs: int):
        """Get scenario instance by name."""
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_class = SCENARIOS[scenario_name]
        scenario = scenario_class(self.config, num_uavs)
        scenario.setup()
        return scenario

    def run_single(self, algorithm_name: str, scenario_name: str, num_uavs: int):
        """Run single algorithm on single scenario."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Running: {algorithm_name} on {scenario_name} with {num_uavs} UAVs")
        self.logger.info(f"{'='*80}\n")

        # Create algorithm and scenario
        algorithm = self.get_algorithm(algorithm_name, num_uavs)
        scenario = self.get_scenario(scenario_name, num_uavs)

        # Get optimization parameters
        algo_config = self.config.get('algorithms', {}).get(algorithm_name, {})
        max_iterations = algo_config.get('max_iterations', 100)

        # Check if algorithm is multi-objective
        is_multi_objective = algorithm_name.lower() in ['mopso', 'mogwo', 'moaco']

        if is_multi_objective:
            # Run TRUE multi-objective optimization
            self.logger.info("Running TRUE multi-objective optimization...")
            self.logger.info(f"Objectives: {scenario.get_objective_names()}")

            pareto_positions, pareto_objectives, stats = algorithm.optimize_multi_objective(
                objective_function=scenario.evaluate_objectives,
                bounds=scenario.get_bounds(),
                max_iterations=max_iterations
            )

            # Get best solution from Pareto front (min sum for display)
            objective_sums = [np.sum(obj) for obj in pareto_objectives]
            best_idx = np.argmin(objective_sums)
            best_position = pareto_positions[best_idx]
            best_fitness = objective_sums[best_idx]

            self.logger.info(f"\nMulti-Objective Optimization Complete!")
            self.logger.info(f"Pareto Front Size: {stats['pareto_size']}")
            self.logger.info(f"Best Solution (min sum): {best_fitness:.4f}")
            self.logger.info(f"Best Objectives: {pareto_objectives[best_idx]}")
            self.logger.info(f"Total Time: {stats['total_time']:.2f}s")

            # Store Pareto front for later analysis
            stats['pareto_positions'] = pareto_positions
            stats['pareto_objectives'] = pareto_objectives
        else:
            # Run single-objective optimization
            self.logger.info("Running single-objective optimization (weighted-sum)...")

            best_position, best_fitness, stats = algorithm.optimize(
                objective_function=scenario.evaluate_solution,
                bounds=scenario.get_bounds(),
                max_iterations=max_iterations
            )

            self.logger.info(f"\nOptimization Complete!")
            self.logger.info(f"Best Fitness: {best_fitness:.4f}")
            self.logger.info(f"Convergence Iteration: {stats['convergence_iteration']}")
            self.logger.info(f"Total Time: {stats['total_time']:.2f}s")

        # Calculate metrics
        swarm_data = {
            'num_uavs': num_uavs,
            'total_distance': 0,
            'mission_time': stats['total_time'],
            'coverage_ratio': 0,
            'collision_count': 0,
            'min_battery_level': 1.0,
            'avg_battery_level': 1.0,
            'avg_connectivity': 1.0,
            'min_connectivity': 1.0,
            'avg_formation_error': 0,
            'max_formation_error': 0,
            'targets_reached': num_uavs,
            'total_energy': 0,
            'min_obstacle_distance': float('inf'),
            'avg_obstacle_distance': 0,
            'safety_violations': 0,
            'min_separation': 3.0,
            'avg_separation': 10.0,
            'active_uavs': num_uavs,
            'near_miss_count': 0,
        }

        metrics = self.metrics_calculator.calculate_all_metrics(
            algorithm_name=algorithm_name,
            scenario_name=scenario_name,
            optimization_stats=stats,
            swarm_data=swarm_data,
            environment_data={}
        )

        # Print key metrics
        self.logger.info("\nKey Metrics:")
        self.logger.info(f"  Success Rate: {metrics['success_rate']:.2%}")
        self.logger.info(f"  Convergence Speed: {metrics['convergence_speed']:.4f}")
        self.logger.info(f"  Number of Evaluations: {metrics['num_evaluations']}")

        # Visualize results
        if self.config.get('visualization', {}).get('enabled', True):
            self.logger.info("\nGenerating visualizations...")

            # Convergence plot
            self.plotter.plot_convergence(
                {algorithm_name: stats},
                filename=f"{algorithm_name}_{scenario_name}_convergence.png"
            )

            # Plot Pareto front for multi-objective algorithms
            if is_multi_objective and hasattr(algorithm, 'get_pareto_front'):
                pareto_positions, pareto_objectives = algorithm.get_pareto_front()
                if pareto_objectives:
                    self.logger.info(f"  Pareto front size: {len(pareto_objectives)}")
                    self.plotter.plot_pareto_fronts(
                        {algorithm_name: (pareto_positions, pareto_objectives)},
                        filename=f"{algorithm_name}_{scenario_name}_pareto.png"
                    )

            self.logger.info("Visualizations saved to results/plots/")

        return best_position, best_fitness, metrics

    def run_benchmark(self, num_uavs: int = 10):
        """Run complete benchmark across all algorithms and scenarios."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Running Complete Benchmark Suite")
        self.logger.info(f"{'='*80}\n")

        # Get all algorithms and scenarios
        algorithm_names = ['pso', 'gwo', 'aco', 'mopso', 'mogwo', 'moaco']
        scenario_names = list(SCENARIOS.keys())

        algorithms = [self.get_algorithm(name, num_uavs) for name in algorithm_names]
        scenarios = [self.get_scenario(name, num_uavs) for name in scenario_names]

        # Run benchmark
        num_trials = self.config.get('evaluation', {}).get('num_trials', 30)
        benchmark_runner = BenchmarkRunner(self.config)

        results = benchmark_runner.run_benchmark(algorithms, scenarios, num_trials)

        # Statistical analysis
        self.logger.info("\nPerforming statistical analysis...")
        analyzer = StatisticalAnalyzer(
            confidence_level=self.config.get('evaluation', {}).get('confidence_level', 0.95)
        )

        # Separate SO and MO algorithms
        so_algorithm_names = ['pso', 'gwo', 'aco']
        mo_algorithm_names = ['mopso', 'mogwo', 'moaco']

        # Organize results by algorithm (separately for SO and MO)
        so_results_by_algorithm = {}
        mo_results_by_algorithm = {}

        for algo_name in so_algorithm_names:
            algo_results = [r['best_fitness'] for r in results['results']
                          if r['algorithm'].lower().replace('_', '') == algo_name.lower().replace('_', '')]
            so_results_by_algorithm[algo_name] = algo_results
            self.logger.info(f"Found {len(algo_results)} results for SO algorithm: {algo_name}")

        for algo_name in mo_algorithm_names:
            algo_results = [r['best_fitness'] for r in results['results']
                          if r['algorithm'].lower().replace('_', '') == algo_name.lower().replace('_', '')]
            mo_results_by_algorithm[algo_name] = algo_results
            self.logger.info(f"Found {len(algo_results)} results for MO algorithm: {algo_name}")

        # Compare SO algorithms
        if so_results_by_algorithm:
            self.logger.info("\n" + "="*80)
            self.logger.info("SINGLE-OBJECTIVE ALGORITHM RANKINGS:")
            self.logger.info("="*80)
            so_comparison = analyzer.compare_algorithms(so_results_by_algorithm, test_type='wilcoxon')
            so_rankings = analyzer.generate_rankings(so_results_by_algorithm)

            for rank_info in so_rankings['rankings']:
                self.logger.info(f"  {rank_info['rank']}. {rank_info['algorithm'].upper()}: "
                               f"Mean={rank_info['mean']:.4f} (std={rank_info['std']:.4f})")

        # Compare MO algorithms
        if mo_results_by_algorithm:
            self.logger.info("\n" + "="*80)
            self.logger.info("MULTI-OBJECTIVE ALGORITHM RANKINGS:")
            self.logger.info("="*80)
            mo_comparison = analyzer.compare_algorithms(mo_results_by_algorithm, test_type='wilcoxon')
            mo_rankings = analyzer.generate_rankings(mo_results_by_algorithm)

            for rank_info in mo_rankings['rankings']:
                self.logger.info(f"  {rank_info['rank']}. {rank_info['algorithm'].upper()}: "
                               f"Mean={rank_info['mean']:.4f} (std={rank_info['std']:.4f})")

        # Generate visualizations
        self.logger.info("\nGenerating comprehensive visualizations...")

        # Convergence comparison (separated)
        so_stats_by_algo = {}
        mo_stats_by_algo = {}

        for algo_name in so_algorithm_names:
            algo_stats = [r['statistics'] for r in results['results']
                         if r['algorithm'].lower().replace('_', '') == algo_name.lower().replace('_', '')]
            if algo_stats:
                so_stats_by_algo[algo_name] = algo_stats[0]

        for algo_name in mo_algorithm_names:
            algo_stats = [r['statistics'] for r in results['results']
                         if r['algorithm'].lower().replace('_', '') == algo_name.lower().replace('_', '')]
            if algo_stats:
                mo_stats_by_algo[algo_name] = algo_stats[0]

        # Plot separate convergence curves
        if so_stats_by_algo:
            self.plotter.plot_convergence(so_stats_by_algo, filename="benchmark_convergence_SO.png")
        if mo_stats_by_algo:
            self.plotter.plot_convergence(mo_stats_by_algo, filename="benchmark_convergence_MO.png")

        # Metrics comparison (separated)
        if so_results_by_algorithm:
            so_metrics_by_algo = {}
            for algo_name in so_algorithm_names:
                if algo_name in so_results_by_algorithm:
                    so_metrics_by_algo[algo_name] = so_results_by_algorithm[algo_name]

            self.plotter.plot_metrics_comparison(
                {k: [{'best_fitness': v} for v in vals] for k, vals in so_metrics_by_algo.items()},
                'best_fitness',
                filename="fitness_comparison_SO.png"
            )

        if mo_results_by_algorithm:
            mo_metrics_by_algo = {}
            for algo_name in mo_algorithm_names:
                if algo_name in mo_results_by_algorithm:
                    mo_metrics_by_algo[algo_name] = mo_results_by_algorithm[algo_name]

            self.plotter.plot_metrics_comparison(
                {k: [{'best_fitness': v} for v in vals] for k, vals in mo_metrics_by_algo.items()},
                'best_fitness',
                filename="fitness_comparison_MO.png"
            )

        # Statistical comparison heatmaps (separated)
        if so_results_by_algorithm:
            self.plotter.plot_statistical_comparison(
                so_comparison,
                filename="statistical_comparison_SO.png"
            )

        if mo_results_by_algorithm:
            self.plotter.plot_statistical_comparison(
                mo_comparison,
                filename="statistical_comparison_MO.png"
            )

        self.logger.info("\nBenchmark complete! Results saved to results/")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="UAV Swarm Optimization Framework - Benchmark Study"
    )

    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--algorithm', type=str, default='pso',
                       choices=['pso', 'gwo', 'aco', 'mopso', 'mogwo', 'moaco'],
                       help='Algorithm to use')
    parser.add_argument('--scenario', type=str, default='obstacle_avoidance',
                       choices=list(SCENARIOS.keys()),
                       help='Scenario to run')
    parser.add_argument('--num_uavs', type=int, default=10,
                       help='Number of UAVs in swarm')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'benchmark'],
                       help='Run mode: single test or full benchmark')
    parser.add_argument('--use_px4', action='store_true',
                       help='Use PX4 SITL integration')

    args = parser.parse_args()

    # Create simulation
    sim = UAVSwarmSimulation(args.config)

    # Update PX4 config if requested
    if args.use_px4:
        sim.config['px4']['enabled'] = True
        sim.logger.info("PX4 SITL integration enabled")

    # Run simulation
    if args.mode == 'single':
        sim.run_single(args.algorithm, args.scenario, args.num_uavs)
    elif args.mode == 'benchmark':
        sim.run_benchmark(args.num_uavs)

    sim.logger.info("\nSimulation complete!")


if __name__ == '__main__':
    main()
