#!/usr/bin/env python3
"""
Independent single-objective benchmark rerun for Paper A.

This script keeps old result folders untouched and fixes trial independence by:
1. using a fresh algorithm instance for every trial
2. using a fresh scenario instance for every trial
3. assigning a distinct deterministic seed to every algorithm/scenario/trial run
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add repository root for package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.algorithms.aco import ACO
from src.algorithms.gwo import GWO
from src.algorithms.pso import PSO
from src.evaluation.statistical_analysis import StatisticalAnalyzer
from src.scenarios.area_coverage import AreaCoverageScenario
from src.scenarios.dynamic_obstacle_avoidance import DynamicObstacleAvoidanceScenario
from src.scenarios.formation_flight import FormationFlightScenario
from src.scenarios.multi_target_engagement import MultiTargetEngagementScenario
from src.scenarios.obstacle_avoidance import ObstacleAvoidanceScenario
from src.scenarios.target_tracking import TargetTrackingScenario
from src.utils.config_loader import load_config


ALGORITHM_CLASSES = {
    "PSO": PSO,
    "GWO": GWO,
    "ACO": ACO,
}

SCENARIO_CLASSES = {
    "obstacle_avoidance": ObstacleAvoidanceScenario,
    "dynamic_obstacle_avoidance": DynamicObstacleAvoidanceScenario,
    "formation_flight": FormationFlightScenario,
    "area_coverage": AreaCoverageScenario,
    "target_tracking": TargetTrackingScenario,
    "multi_target_engagement": MultiTargetEngagementScenario,
}

WORKER_CONFIG = None
WORKER_NUM_UAVS = None


def derive_trial_seed(base_seed: int, algorithm_name: str, scenario_name: str, trial: int) -> int:
    """Create a stable, distinct seed per algorithm/scenario/trial."""
    seed_material = f"{base_seed}:{algorithm_name}:{scenario_name}:{trial}".encode("utf-8")
    digest = hashlib.sha256(seed_material).hexdigest()
    return int(digest[:8], 16)


def set_trial_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for a trial."""
    random.seed(seed)
    np.random.seed(seed)


def create_algorithm(algorithm_name: str, config: dict, num_uavs: int):
    """Instantiate one algorithm by name."""
    return ALGORITHM_CLASSES[algorithm_name](config=config, num_uavs=num_uavs)


def create_scenario(scenario_name: str, config: dict, num_uavs: int):
    """Instantiate one scenario by name."""
    return SCENARIO_CLASSES[scenario_name](config=config, num_uavs=num_uavs)


def initialize_worker(config: dict, num_uavs: int):
    """Install shared config in each worker process."""
    global WORKER_CONFIG, WORKER_NUM_UAVS
    WORKER_CONFIG = config
    WORKER_NUM_UAVS = num_uavs


def run_single_trial(algorithm_name: str, scenario_name: str, trial: int, seed: int):
    """Execute one independent optimization trial."""
    set_trial_seed(seed)

    algorithm = create_algorithm(algorithm_name, WORKER_CONFIG, WORKER_NUM_UAVS)
    scenario = create_scenario(scenario_name, WORKER_CONFIG, WORKER_NUM_UAVS)
    scenario.reset()

    max_iterations = WORKER_CONFIG["algorithms"][algorithm_name.lower()]["max_iterations"]

    start_time = time.time()
    best_position, best_fitness, stats = algorithm.optimize(
        objective_function=scenario.evaluate_solution,
        bounds=scenario.get_bounds(),
        max_iterations=max_iterations,
    )
    run_time = time.time() - start_time

    stats["best_fitness"] = float(best_fitness)
    stats["total_time"] = run_time

    return {
        "algorithm": algorithm_name,
        "scenario": scenario_name,
        "trial": trial,
        "seed": seed,
        "best_fitness": float(best_fitness),
        "best_position": best_position.tolist() if isinstance(best_position, np.ndarray) else best_position,
        "run_time": run_time,
        "statistics": stats,
        "metrics": {
            "completion_time": run_time,
            "success_rate": 1.0,
        },
        "is_multi_objective": False,
    }


def generate_comparison_plots(results, output_dir):
    """Generate comparison plots from the independent rerun results."""
    mpl_dir = tempfile.mkdtemp(prefix="paper_a_mpl_")
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\nGenerating comparison plots...")

    algorithms = list(ALGORITHM_CLASSES.keys())
    scenarios = list(SCENARIO_CLASSES.keys())
    colors = {"PSO": "#1f77b4", "GWO": "#ff7f0e", "ACO": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(scenarios))
    width = 0.25

    for i, algorithm_name in enumerate(algorithms):
        means = []
        stds = []

        for scenario_name in scenarios:
            scenario_results = [
                record["best_fitness"] for record in results
                if record["algorithm"] == algorithm_name and record["scenario"] == scenario_name
            ]
            means.append(float(np.mean(scenario_results)) if scenario_results else 0.0)
            stds.append(float(np.std(scenario_results)) if scenario_results else 0.0)

        ax.bar(
            x + i * width,
            means,
            width,
            label=algorithm_name,
            yerr=stds,
            capsize=5,
            color=colors[algorithm_name],
        )

    ax.set_xlabel("Scenario", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Fitness (Lower is Better)", fontsize=12, fontweight="bold")
    ax.set_title("Independent Single-Objective Algorithm Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/algorithm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    for scenario_name in scenarios:
        fig, ax = plt.subplots(figsize=(10, 6))
        scenario_results = [record for record in results if record["scenario"] == scenario_name]

        data_to_plot = []
        labels = []
        for algorithm_name in algorithms:
            algorithm_data = [
                record["best_fitness"] for record in scenario_results
                if record["algorithm"] == algorithm_name
            ]
            if algorithm_data:
                data_to_plot.append(algorithm_data)
                labels.append(algorithm_name)

        boxplot = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        for patch, algorithm_name in zip(boxplot["boxes"], labels):
            patch.set_facecolor(colors[algorithm_name])
            patch.set_alpha(0.7)

        ax.set_xlabel("Algorithm", fontsize=11, fontweight="bold")
        ax.set_ylabel("Fitness Distribution", fontsize=11, fontweight="bold")
        ax.set_title(f"{scenario_name} - Independent Trial Comparison", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{scenario_name}_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("All comparison plots generated!")


def json_default(value):
    """Convert NumPy scalars and arrays to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def calculate_aggregate_stats(algorithm_name: str, scenario_name: str, results_for_pair):
    """Compute aggregate statistics for one algorithm-scenario combination."""
    fitness_values = [record["best_fitness"] for record in results_for_pair]
    run_times = [record["run_time"] for record in results_for_pair]

    return {
        "algorithm": algorithm_name,
        "scenario": scenario_name,
        "mean_fitness": float(np.mean(fitness_values)),
        "std_fitness": float(np.std(fitness_values)),
        "median_fitness": float(np.median(fitness_values)),
        "best_fitness": float(np.min(fitness_values)),
        "worst_fitness": float(np.max(fitness_values)),
        "mean_time": float(np.mean(run_times)),
        "std_time": float(np.std(run_times)),
    }


def generate_summary(results):
    """Summarize independent rerun results."""
    summary = {
        "total_runs": len(results),
        "algorithms": list(ALGORITHM_CLASSES.keys()),
        "scenarios": list(SCENARIO_CLASSES.keys()),
        "by_algorithm": {},
        "by_scenario": {},
    }

    for algorithm_name in summary["algorithms"]:
        algorithm_results = [record for record in results if record["algorithm"] == algorithm_name]
        fitness_values = [record["best_fitness"] for record in algorithm_results]
        summary["by_algorithm"][algorithm_name] = {
            "mean_fitness": float(np.mean(fitness_values)),
            "std_fitness": float(np.std(fitness_values)),
            "best_fitness": float(np.min(fitness_values)),
        }

    for scenario_name in summary["scenarios"]:
        scenario_results = [record for record in results if record["scenario"] == scenario_name]
        fitness_values = [record["best_fitness"] for record in scenario_results]
        summary["by_scenario"][scenario_name] = {
            "mean_fitness": float(np.mean(fitness_values)),
            "std_fitness": float(np.std(fitness_values)),
            "best_fitness": float(np.min(fitness_values)),
        }

    return summary


def build_statistical_inputs(results):
    """Convert detailed trial results to statistical input structures."""
    scenario_trial_fitness = {}
    scenario_mean_fitness = {}

    for scenario_name in SCENARIO_CLASSES:
        scenario_trial_fitness[scenario_name] = {}
        scenario_mean_fitness[scenario_name] = {}

        for algorithm_name in ALGORITHM_CLASSES:
            values = [
                record["best_fitness"] for record in results
                if record["scenario"] == scenario_name and record["algorithm"] == algorithm_name
            ]
            scenario_trial_fitness[scenario_name][algorithm_name] = values
            scenario_mean_fitness[scenario_name][algorithm_name] = float(np.mean(values))

    return scenario_trial_fitness, scenario_mean_fitness


def save_wilcoxon_csv(comparisons, csv_path):
    """Save exact Wilcoxon results to CSV."""
    fieldnames = [
        "scenario",
        "algorithm_a",
        "algorithm_b",
        "statistic",
        "raw_p_value",
        "corrected_p_value",
        "significant_raw",
        "significant_corrected",
    ]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({
                key: comparison.get(key)
                for key in fieldnames
            })


def save_friedman_csv(scenario_ranks, mean_ranks, csv_path):
    """Save per-scenario Friedman ranks and mean ranks to CSV."""
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["scenario", "PSO", "GWO", "ACO"])
        for scenario_name, ranks in scenario_ranks.items():
            writer.writerow([
                scenario_name,
                ranks.get("PSO"),
                ranks.get("GWO"),
                ranks.get("ACO"),
            ])
        writer.writerow([])
        writer.writerow(["mean_rank", mean_ranks.get("PSO"), mean_ranks.get("GWO"), mean_ranks.get("ACO")])


def run_independent_benchmark(
    config,
    output_dir: Path,
    num_trials: int,
    base_seed: int,
    skip_plots: bool,
    worker_count: int | None,
):
    """Run a fresh, independent-trial benchmark and save new outputs."""
    num_uavs = config.get("uav", {}).get("num_uavs", 10)
    all_results = []

    total_runs = len(ALGORITHM_CLASSES) * len(SCENARIO_CLASSES) * num_trials
    progress = tqdm(total=total_runs, desc="Independent Benchmark Progress")
    trial_specs = []
    for algorithm_name in ALGORITHM_CLASSES:
        for scenario_name in SCENARIO_CLASSES:
            for trial in range(num_trials):
                seed = derive_trial_seed(base_seed, algorithm_name, scenario_name, trial)
                trial_specs.append((algorithm_name, scenario_name, trial, seed))

    if worker_count is None:
        worker_count = max(1, min(os.cpu_count() or 1, 6))
    else:
        worker_count = max(1, worker_count)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=initialize_worker,
        initargs=(config, num_uavs),
    ) as executor:
        future_map = {
            executor.submit(run_single_trial, algorithm_name, scenario_name, trial, seed): (
                algorithm_name,
                scenario_name,
                trial,
            )
            for algorithm_name, scenario_name, trial, seed in trial_specs
        }

        for future in as_completed(future_map):
            result = future.result()
            all_results.append(result)
            progress.update(1)

    progress.close()
    all_results.sort(key=lambda record: (record["algorithm"], record["scenario"], record["trial"]))

    benchmark_stats = []
    for (algorithm_name, scenario_name), grouped_records in groupby(
        all_results,
        key=lambda record: (record["algorithm"], record["scenario"]),
    ):
        benchmark_stats.append(
            calculate_aggregate_stats(algorithm_name, scenario_name, list(grouped_records))
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"benchmark_{timestamp}.json"
    stats_path = output_dir / f"benchmark_stats_{timestamp}.json"

    with open(results_path, "w") as results_file:
        json.dump(all_results, results_file, indent=2, default=json_default)

    with open(stats_path, "w") as stats_file:
        json.dump(benchmark_stats, stats_file, indent=2, default=json_default)

    summary = generate_summary(all_results)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2, default=json_default)

    analyzer = StatisticalAnalyzer()
    scenario_trial_fitness, scenario_mean_fitness = build_statistical_inputs(all_results)
    wilcoxon_results = analyzer.compare_algorithms(scenario_trial_fitness, test_type="wilcoxon")
    friedman_results = analyzer.compare_algorithms(scenario_mean_fitness, test_type="friedman")

    analysis_payload = {
        "base_seed": base_seed,
        "num_trials": num_trials,
        "wilcoxon": wilcoxon_results,
        "friedman": friedman_results,
    }
    analysis_path = output_dir / "analysis_summary.json"
    with open(analysis_path, "w") as analysis_file:
        json.dump(analysis_payload, analysis_file, indent=2, default=json_default)

    save_wilcoxon_csv(wilcoxon_results["comparisons"], output_dir / "wilcoxon_exact_pvalues.csv")
    save_friedman_csv(
        friedman_results["scenario_ranks"],
        friedman_results["mean_ranks"],
        output_dir / "friedman_ranks.csv",
    )

    if not skip_plots:
        generate_comparison_plots(all_results, output_dir)

    return {
        "results_path": results_path,
        "stats_path": stats_path,
        "summary_path": summary_path,
        "analysis_path": analysis_path,
        "summary": summary,
        "wilcoxon": wilcoxon_results,
        "friedman": friedman_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Independent single-objective benchmark rerun for Paper A"
    )
    parser.add_argument("--trials", type=int, default=30, help="Number of trials per algorithm-scenario combination")
    parser.add_argument("--output", type=str, default="results_paper_a_independent", help="Output directory root")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--base-seed", type=int, default=20260528, help="Base seed used to derive per-trial seeds")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--workers", type=int, default=None, help="Reserved for future manual worker override")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PAPER A: INDEPENDENT SINGLE-OBJECTIVE BENCHMARK RERUN")
    print("=" * 80)
    print(f"Trials per combination: {args.trials}")
    print(f"Base seed: {args.base_seed}")
    print(f"Output directory: {output_dir}")

    report = run_independent_benchmark(
        config=config,
        output_dir=output_dir,
        num_trials=args.trials,
        base_seed=args.base_seed,
        skip_plots=args.skip_plots,
        worker_count=args.workers,
    )

    print("\n" + "=" * 80)
    print("INDEPENDENT RERUN SUMMARY")
    print("=" * 80)

    for algorithm_name in ALGORITHM_CLASSES:
        stats = report["summary"]["by_algorithm"][algorithm_name]
        print(f"\n{algorithm_name}:")
        print(f"  Mean Fitness: {stats['mean_fitness']:.4f} ± {stats['std_fitness']:.4f}")
        print(f"  Best Fitness: {stats['best_fitness']:.4f}")

    print("\nFriedman mean ranks:")
    for algorithm_name in ALGORITHM_CLASSES:
        print(f"  {algorithm_name}: {report['friedman']['mean_ranks'][algorithm_name]:.6f}")

    print("\nFiles saved:")
    print(f"  Detailed results: {report['results_path']}")
    print(f"  Aggregate stats: {report['stats_path']}")
    print(f"  Summary: {report['summary_path']}")
    print(f"  Analysis: {report['analysis_path']}")


if __name__ == "__main__":
    main()
