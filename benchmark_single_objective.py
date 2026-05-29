#!/usr/bin/env python3
"""
Paper A: Single-Objective Swarm Optimization Benchmark
Algorithms: PSO, GWO, ACO
Scenarios: 6 mission scenarios (obstacle_avoidance, dynamic_obstacle_avoidance,
           formation_flight, area_coverage, target_tracking, multi_target_engagement)

Usage:
    python benchmark_single_objective.py [--trials NUM] [--output DIR]
"""

import argparse
import sys
import os
import json
import importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for proper package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STATS_FILE = (
    Path(__file__).resolve().parent /
    "results_paper_a" /
    "20251228_001329" /
    "so_benchmark_stats_20251228_001354.json"
)
DEFAULT_RAW_RESULTS_FILE = (
    Path(__file__).resolve().parent /
    "results_paper_a" /
    "20251228_001329" /
    "so_benchmark_20251228_001354.json"
)
STATISTICAL_ANALYSIS_FILE = (
    Path(__file__).resolve().parent /
    "src" /
    "evaluation" /
    "statistical_analysis.py"
)


_STATISTICAL_ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "paper_a_statistical_analysis",
    STATISTICAL_ANALYSIS_FILE
)
_STATISTICAL_ANALYSIS_MODULE = importlib.util.module_from_spec(_STATISTICAL_ANALYSIS_SPEC)
_STATISTICAL_ANALYSIS_SPEC.loader.exec_module(_STATISTICAL_ANALYSIS_MODULE)
StatisticalAnalyzer = _STATISTICAL_ANALYSIS_MODULE.StatisticalAnalyzer


def create_algorithm_instances(config, num_uavs):
    """Create instances of single-objective algorithms."""
    from src.algorithms.pso import PSO
    from src.algorithms.gwo import GWO
    from src.algorithms.aco import ACO

    algorithms = [
        PSO(num_uavs=num_uavs, config=config),
        GWO(num_uavs=num_uavs, config=config),
        ACO(num_uavs=num_uavs, config=config),
    ]
    return algorithms


def create_scenario_instances(config):
    """Create instances of 6 mission scenarios."""
    from src.scenarios.obstacle_avoidance import ObstacleAvoidanceScenario
    from src.scenarios.dynamic_obstacle_avoidance import DynamicObstacleAvoidanceScenario
    from src.scenarios.formation_flight import FormationFlightScenario
    from src.scenarios.area_coverage import AreaCoverageScenario
    from src.scenarios.target_tracking import TargetTrackingScenario
    from src.scenarios.multi_target_engagement import MultiTargetEngagementScenario

    num_uavs = config.get('uav', {}).get('num_uavs', 10)

    scenarios = [
        ObstacleAvoidanceScenario(config=config, num_uavs=num_uavs),
        DynamicObstacleAvoidanceScenario(config=config, num_uavs=num_uavs),
        FormationFlightScenario(config=config, num_uavs=num_uavs),
        AreaCoverageScenario(config=config, num_uavs=num_uavs),
        TargetTrackingScenario(config=config, num_uavs=num_uavs),
        MultiTargetEngagementScenario(config=config, num_uavs=num_uavs),
    ]
    return scenarios


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots for single-objective algorithms."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    print("\nGenerating comparison plots...")

    # Get unique algorithms and scenarios
    algorithms = list(set(r['algorithm'] for r in results))
    scenarios = list(set(r['scenario'] for r in results))

    # 1. Algorithm Comparison (Mean Fitness)
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(scenarios))
    width = 0.25

    colors = {'PSO': '#1f77b4', 'GWO': '#ff7f0e', 'ACO': '#2ca02c'}

    for i, algo in enumerate(sorted(algorithms)):
        means = []
        stds = []
        for scenario in scenarios:
            scenario_results = [r['best_fitness'] for r in results
                                if r['algorithm'] == algo and r['scenario'] == scenario]
            if scenario_results:
                means.append(np.mean(scenario_results))
                stds.append(np.std(scenario_results))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, label=algo, yerr=stds, capsize=5,
               color=colors.get(algo, '#888888'))

    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Fitness (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Single-Objective Algorithm Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/algorithm_comparison.png")

    # 2. Per-scenario boxplots
    for scenario in scenarios:
        fig, ax = plt.subplots(figsize=(10, 6))
        scenario_results = [r for r in results if r['scenario'] == scenario]

        data_to_plot = []
        labels = []
        for algo in sorted(algorithms):
            algo_data = [r['best_fitness'] for r in scenario_results if r['algorithm'] == algo]
            if algo_data:
                data_to_plot.append(algo_data)
                labels.append(algo)

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, algo in zip(bp['boxes'], labels):
            patch.set_facecolor(colors.get(algo, '#888888'))
            patch.set_alpha(0.7)

        ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax.set_ylabel('Fitness Distribution', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario} - Algorithm Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{scenario}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("All comparison plots generated!")


def _load_json_records(json_path):
    """Load a JSON file that contains a list of benchmark records."""
    path = Path(json_path)
    with open(path, 'r') as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(records).__name__}")

    return records


def report_friedman_rank_aggregation(stats_file):
    """Report Friedman mean-rank aggregation across scenarios."""
    print("\n" + "=" * 80)
    print("FRIEDMAN RANK AGGREGATION")
    print("=" * 80)

    stats_records = _load_json_records(stats_file)
    scenario_mean_fitness = {}

    for record in stats_records:
        scenario = record['scenario']
        algorithm = record['algorithm']
        mean_fitness = float(record['mean_fitness'])
        scenario_mean_fitness.setdefault(scenario, {})[algorithm] = mean_fitness

    analyzer = StatisticalAnalyzer()
    friedman_results = analyzer.compare_algorithms(
        scenario_mean_fitness,
        test_type='friedman'
    )

    scenario_ranks = friedman_results['scenario_ranks']
    mean_ranks = friedman_results['mean_ranks']

    print(f"Stats file: {stats_file}")
    print(f"Friedman statistic: {friedman_results['statistic']:.6f}")
    print(f"Friedman p-value: {friedman_results['p_value']:.6f}")
    print("\nPer-scenario ranks (1 = best / lowest mean fitness):")
    for scenario, ranks in scenario_ranks.items():
        rank_text = ", ".join(
            f"{algorithm}={rank:.2f}"
            for algorithm, rank in ranks.items()
        )
        print(f"  {scenario}: {rank_text}")

    print("\nMean Friedman ranks across the 6 scenarios:")
    for algorithm in ['PSO', 'GWO', 'ACO']:
        if algorithm in mean_ranks:
            print(f"  {algorithm}: {mean_ranks[algorithm]:.6f}")

    pso_rank = mean_ranks['PSO']
    gwo_ratio = mean_ranks['GWO'] / pso_rank
    aco_ratio = mean_ranks['ACO'] / pso_rank

    print("\nRank-based performance ratios:")
    print(f"  rank(GWO) / rank(PSO) = {gwo_ratio:.6f}")
    print(f"  rank(ACO) / rank(PSO) = {aco_ratio:.6f}")


def report_holm_bonferroni_wilcoxon(raw_results_file):
    """Report Holm-Bonferroni corrected Wilcoxon tests across scenarios."""
    print("\n" + "=" * 80)
    print("HOLM-BONFERRONI CORRECTED WILCOXON TESTS")
    print("=" * 80)

    raw_results = _load_json_records(raw_results_file)
    scenario_trial_fitness = {}

    for record in raw_results:
        scenario = record['scenario']
        algorithm = record['algorithm']
        best_fitness = float(record['best_fitness'])
        scenario_trial_fitness.setdefault(scenario, {}).setdefault(algorithm, []).append(best_fitness)

    analyzer = StatisticalAnalyzer()
    wilcoxon_results = analyzer.compare_algorithms(
        scenario_trial_fitness,
        test_type='wilcoxon'
    )

    print(f"Raw results file: {raw_results_file}")
    print(f"Total pairwise scenario comparisons: {len(wilcoxon_results['comparisons'])}")
    print("\nAll 18 pairwise comparisons:")
    for comparison in wilcoxon_results['comparisons']:
        significance_label = "significant" if comparison['significant_corrected'] else "not significant"
        print(
            "  "
            f"{comparison['scenario']}: "
            f"{comparison['algorithm_a']} vs {comparison['algorithm_b']} -> "
            f"raw p = {comparison['raw_p_value']:.16g}, "
            f"Holm-corrected p = {comparison['corrected_p_value']:.16g}, "
            f"{significance_label} after correction"
        )

    significant_after_correction = wilcoxon_results['significant_after_correction']
    print("\nSummary of comparisons that remain significant after Holm-Bonferroni:")
    if significant_after_correction:
        for comparison in significant_after_correction:
            print(
                "  "
                f"{comparison['scenario']}: "
                f"{comparison['algorithm_a']} vs {comparison['algorithm_b']} "
                f"(corrected p = {comparison['corrected_p_value']:.16g})"
            )
    else:
        print("  None")


def report_safety_radius_check():
    """Report the numeric safety-radius values used in the current code."""
    print("\n" + "=" * 80)
    print("READ-ONLY SAFETY RADIUS CHECK")
    print("=" * 80)

    print("Environment / obstacle-layer values:")
    print("  src/environment/environment.py -> no collision safety radius variable is defined here")
    print("  src/environment/obstacles.py -> Obstacle.contains_point(..., safety_margin=0.0)")
    print("  src/environment/obstacles.py -> ObstacleManager.is_position_valid(..., safety_margin=1.0)")

    print("\nActual obstacle-distance thresholds used during fitness evaluation:")
    print("  src/scenarios/obstacle_avoidance.py -> safety_distance = 5.0")
    print("  src/scenarios/formation_flight.py -> safety_distance = 5.0")
    print("  src/scenarios/dynamic_obstacle_avoidance.py -> safety_distance = 6.0")

    print("\nCollision radius used in the UAV simulation layer:")
    print("  src/simulation/uav.py -> collision_radius = 0.5")


def main():
    parser = argparse.ArgumentParser(
        description='Paper A: Single-Objective Swarm Optimization Benchmark'
    )
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of trials per algorithm-scenario combination (default: 30)')
    parser.add_argument('--output', type=str, default='results_paper_a',
                        help='Output directory for results')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip the benchmark run and analyze saved Paper A outputs only')
    parser.add_argument('--stats-file', type=str, default=str(DEFAULT_STATS_FILE),
                        help='Path to saved aggregate single-objective statistics JSON')
    parser.add_argument('--raw-results-file', type=str, default=str(DEFAULT_RAW_RESULTS_FILE),
                        help='Path to saved detailed single-objective benchmark JSON')
    args = parser.parse_args()

    if args.analysis_only:
        report_friedman_rank_aggregation(args.stats_file)
        report_holm_bonferroni_wilcoxon(args.raw_results_file)
        report_safety_radius_check()
        return

    from src.utils.config_loader import load_config
    from src.evaluation.benchmarking import BenchmarkRunner

    print("=" * 80)
    print("PAPER A: SINGLE-OBJECTIVE SWARM OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print("\nAlgorithms: PSO, GWO, ACO")
    print("Scenarios: 6 mission scenarios")
    print(f"Trials per combination: {args.trials}")

    # Load config
    print(f"\nLoading config from {args.config}...")
    config = load_config(args.config)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Get number of UAVs
    num_uavs = config.get('uav', {}).get('num_uavs', 10)
    print(f"Number of UAVs: {num_uavs}")

    # Create algorithm and scenario instances
    print("\nInitializing algorithms...")
    algorithms = create_algorithm_instances(config, num_uavs)
    print(f"Algorithms: {[algo.__class__.__name__ for algo in algorithms]}")

    print("\nInitializing scenarios...")
    scenarios = create_scenario_instances(config)
    print(f"Scenarios: {[scenario.get_name() for scenario in scenarios]}")

    # Run benchmark
    total_runs = len(algorithms) * len(scenarios) * args.trials
    print(f"\nRunning benchmark...")
    print(f"Total runs: {len(algorithms)} algorithms × {len(scenarios)} scenarios × {args.trials} trials = {total_runs}")

    benchmark_runner = BenchmarkRunner(config, results_dir=str(output_dir))
    benchmark_data = benchmark_runner.run_benchmark(
        algorithms=algorithms,
        scenarios=scenarios,
        num_trials=args.trials
    )

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    generate_comparison_plots(benchmark_data['results'], output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    summary = benchmark_data['summary']
    print(f"\nTotal runs completed: {summary['total_runs']}")

    for algo in ['PSO', 'GWO', 'ACO']:
        if algo in summary['by_algorithm']:
            stats = summary['by_algorithm'][algo]
            print(f"\n{algo}:")
            print(f"  Mean Fitness: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
            print(f"  Best Fitness: {stats['best_fitness']:.4f}")

    print("\n--- By Scenario ---")
    for scenario in summary['scenarios']:
        stats = summary['by_scenario'][scenario]
        print(f"\n{scenario}:")
        print(f"  Mean Fitness: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
