"""Pareto front visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
from mpl_toolkits.mplot3d import Axes3D


def plot_pareto_2d(
    pareto_fronts: Dict[str, List[np.ndarray]],
    objective_names: List[str],
    title: str = "Pareto Front Comparison",
    save_path: Optional[str] = None
):
    """
    Plot 2D Pareto fronts for multiple algorithms.

    Args:
        pareto_fronts: Dict mapping algorithm names to lists of objective vectors
        objective_names: Names of the two objectives
        title: Plot title
        save_path: Optional path to save the figure
    """
    if len(objective_names) != 2:
        raise ValueError("2D plot requires exactly 2 objectives")

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, (alg_name, objectives) in enumerate(pareto_fronts.items()):
        if len(objectives) == 0:
            continue

        objectives_array = np.array(objectives)
        ax.scatter(
            objectives_array[:, 0],
            objectives_array[:, 1],
            label=alg_name,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(objective_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(objective_names[1], fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D Pareto plot to {save_path}")

    return fig


def plot_pareto_3d(
    pareto_fronts: Dict[str, List[np.ndarray]],
    objective_names: List[str],
    title: str = "3D Pareto Front Comparison",
    save_path: Optional[str] = None
):
    """
    Plot 3D Pareto fronts for multiple algorithms.

    Args:
        pareto_fronts: Dict mapping algorithm names to lists of objective vectors
        objective_names: Names of the three objectives
        title: Plot title
        save_path: Optional path to save the figure
    """
    if len(objective_names) != 3:
        raise ValueError("3D plot requires exactly 3 objectives")

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, (alg_name, objectives) in enumerate(pareto_fronts.items()):
        if len(objectives) == 0:
            continue

        objectives_array = np.array(objectives)
        ax.scatter(
            objectives_array[:, 0],
            objectives_array[:, 1],
            objectives_array[:, 2],
            label=alg_name,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(objective_names[0], fontsize=11, fontweight='bold')
    ax.set_ylabel(objective_names[1], fontsize=11, fontweight='bold')
    ax.set_zlabel(objective_names[2], fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D Pareto plot to {save_path}")

    return fig


def plot_parallel_coordinates(
    pareto_fronts: Dict[str, List[np.ndarray]],
    objective_names: List[str],
    title: str = "Parallel Coordinates - Pareto Fronts",
    save_path: Optional[str] = None
):
    """
    Plot parallel coordinates for Pareto fronts with many objectives.

    Args:
        pareto_fronts: Dict mapping algorithm names to lists of objective vectors
        objective_names: Names of all objectives
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    n_objectives = len(objective_names)
    x_positions = np.arange(n_objectives)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (alg_name, objectives) in enumerate(pareto_fronts.items()):
        if len(objectives) == 0:
            continue

        objectives_array = np.array(objectives)

        # Normalize each objective to [0, 1] for better visualization
        normalized = np.zeros_like(objectives_array)
        for i in range(n_objectives):
            obj_min = objectives_array[:, i].min()
            obj_max = objectives_array[:, i].max()
            if obj_max > obj_min:
                normalized[:, i] = (objectives_array[:, i] - obj_min) / (obj_max - obj_min)
            else:
                normalized[:, i] = 0.5

        # Plot lines for each solution
        for solution in normalized:
            ax.plot(
                x_positions,
                solution,
                color=colors[idx % len(colors)],
                alpha=0.3,
                linewidth=1.5
            )

        # Plot one dummy line for legend
        ax.plot(
            [],
            [],
            color=colors[idx % len(colors)],
            label=alg_name,
            linewidth=3,
            alpha=0.7
        )

    # Set x-ticks to objective names
    ax.set_xticks(x_positions)
    ax.set_xticklabels(objective_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Normalized Objective Value', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parallel coordinates plot to {save_path}")

    return fig


def plot_objective_distribution(
    pareto_fronts: Dict[str, List[np.ndarray]],
    objective_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot distribution of each objective across algorithms.

    Args:
        pareto_fronts: Dict mapping algorithm names to lists of objective vectors
        objective_names: Names of all objectives
        save_path: Optional path to save the figure
    """
    n_objectives = len(objective_names)
    n_algorithms = len(pareto_fronts)

    fig, axes = plt.subplots(1, n_objectives, figsize=(5 * n_objectives, 6))
    if n_objectives == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for obj_idx, (ax, obj_name) in enumerate(zip(axes, objective_names)):
        positions = []
        data_to_plot = []
        labels = []

        for alg_idx, (alg_name, objectives) in enumerate(pareto_fronts.items()):
            if len(objectives) == 0:
                continue

            objectives_array = np.array(objectives)
            objective_values = objectives_array[:, obj_idx]

            positions.append(alg_idx + 1)
            data_to_plot.append(objective_values)
            labels.append(alg_name)

        # Create violin plot
        parts = ax.violinplot(
            data_to_plot,
            positions=positions,
            showmeans=True,
            showmedians=True
        )

        # Color the violins
        for pc, color in zip(parts['bodies'], colors[:len(data_to_plot)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Objective Value', fontsize=11, fontweight='bold')
        ax.set_title(obj_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved objective distribution plot to {save_path}")

    return fig


def plot_convergence_with_hypervolume(
    convergence_data: Dict[str, List[float]],
    title: str = "Hypervolume Convergence",
    save_path: Optional[str] = None
):
    """
    Plot hypervolume convergence over iterations.

    Args:
        convergence_data: Dict mapping algorithm names to hypervolume values over iterations
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for idx, (alg_name, hv_values) in enumerate(convergence_data.items()):
        iterations = np.arange(len(hv_values))
        ax.plot(
            iterations,
            hv_values,
            label=alg_name,
            color=colors[idx % len(colors)],
            linewidth=2,
            marker='o',
            markevery=max(1, len(hv_values) // 10),
            markersize=6
        )

    ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hypervolume', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")

    return fig


def create_pareto_report(
    pareto_fronts: Dict[str, List[np.ndarray]],
    objective_names: List[str],
    metrics: Dict[str, Dict[str, float]],
    scenario_name: str,
    output_dir: str
):
    """
    Create comprehensive Pareto front analysis report with all plots.

    Args:
        pareto_fronts: Dict mapping algorithm names to lists of objective vectors
        objective_names: Names of all objectives
        metrics: Dict of metrics for each algorithm
        scenario_name: Name of the scenario
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    n_objectives = len(objective_names)

    # 2D Pareto plot (if 2 objectives)
    if n_objectives == 2:
        plot_pareto_2d(
            pareto_fronts,
            objective_names,
            title=f"Pareto Front - {scenario_name}",
            save_path=f"{output_dir}/{scenario_name}_pareto_2d.png"
        )

    # 3D Pareto plot (if 3 objectives)
    if n_objectives == 3:
        plot_pareto_3d(
            pareto_fronts,
            objective_names,
            title=f"3D Pareto Front - {scenario_name}",
            save_path=f"{output_dir}/{scenario_name}_pareto_3d.png"
        )

    # Parallel coordinates (for any number of objectives)
    if n_objectives >= 2:
        plot_parallel_coordinates(
            pareto_fronts,
            objective_names,
            title=f"Parallel Coordinates - {scenario_name}",
            save_path=f"{output_dir}/{scenario_name}_parallel.png"
        )

    # Objective distributions
    plot_objective_distribution(
        pareto_fronts,
        objective_names,
        save_path=f"{output_dir}/{scenario_name}_distributions.png"
    )

    # Create metrics table
    _create_metrics_table(metrics, scenario_name, output_dir)

    print(f"\nPareto analysis report created in {output_dir}/")


def _create_metrics_table(
    metrics: Dict[str, Dict[str, float]],
    scenario_name: str,
    output_dir: str
):
    """Create a table visualization of metrics."""
    fig, ax = plt.subplots(figsize=(12, max(4, len(metrics) * 0.8)))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    algorithms = list(metrics.keys())
    metric_names = list(metrics[algorithms[0]].keys()) if algorithms else []

    table_data = []
    for alg in algorithms:
        row = [alg]
        for metric in metric_names:
            value = metrics[alg].get(metric, 0.0)
            if isinstance(value, float):
                row.append(f"{value:.4f}")
            else:
                row.append(str(value))
        table_data.append(row)

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Algorithm'] + metric_names,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] + [0.12] * len(metric_names)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(metric_names) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(algorithms) + 1):
        for j in range(len(metric_names) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.title(f"Multi-Objective Metrics - {scenario_name}", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    save_path = f"{output_dir}/{scenario_name}_metrics_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics table to {save_path}")

    plt.close()
