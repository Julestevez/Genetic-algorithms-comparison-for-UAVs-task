"""Plotting utilities for visualization."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Plotter:
    """Generate plots and visualizations."""

    def __init__(self, results_dir: str = "results/plots"):
        """Initialize plotter."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_convergence(self, stats_by_algorithm: Dict[str, Dict], filename: str = "convergence.png"):
        """Plot convergence curves for algorithms."""
        # Separate SO and MO algorithms
        so_algorithms = {k: v for k, v in stats_by_algorithm.items()
                        if k.lower() not in ['mopso', 'mogwo', 'moaco']}
        mo_algorithms = {k: v for k, v in stats_by_algorithm.items()
                        if k.lower() in ['mopso', 'mogwo', 'moaco']}

        # Create separate plots for SO and MO
        if so_algorithms:
            self._plot_convergence_group(so_algorithms, "Single-Objective",
                                        filename.replace('.png', '_SO.png'))

        if mo_algorithms:
            self._plot_convergence_group(mo_algorithms, "Multi-Objective",
                                        filename.replace('.png', '_MO.png'))

    def _plot_convergence_group(self, stats_by_algorithm: Dict[str, Dict],
                                title_prefix: str, filename: str):
        """Plot convergence curves for a group of algorithms."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for algo_name, stats in stats_by_algorithm.items():
            if 'best_fitness_history' in stats:
                history = stats['best_fitness_history']
                iterations = range(len(history))

                ax1.plot(iterations, history, label=algo_name.upper(), linewidth=2, alpha=0.8)
                ax2.semilogy(iterations, history, label=algo_name.upper(), linewidth=2, alpha=0.8)

        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Best Fitness', fontsize=12)
        ax1.set_title(f'{title_prefix} - Convergence Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Best Fitness (log scale)', fontsize=12)
        ax2.set_title(f'{title_prefix} - Convergence (Log Scale)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_trajectory_3d(self, trajectories: Dict[str, np.ndarray],
                          obstacles: List[Dict] = None,
                          filename: str = "trajectory_3d.html"):
        """Plot 3D trajectories using Plotly."""
        fig = go.Figure()

        # Plot UAV trajectories
        for uav_id, traj in trajectories.items():
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode='lines+markers',
                name=f'UAV {uav_id}',
                line=dict(width=3),
                marker=dict(size=2)
            ))

        # Plot obstacles
        if obstacles:
            for obs in obstacles:
                pos = obs['position']
                size = obs['size']

                # Create box vertices
                vertices = self._create_box_vertices(pos, size)

                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    opacity=0.3,
                    color='red',
                    name='Obstacle'
                ))

        fig.update_layout(
            title='UAV Swarm 3D Trajectories',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            showlegend=True
        )

        fig.write_html(str(self.results_dir / filename))

    def _create_box_vertices(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Create vertices for a 3D box."""
        half_size = size / 2
        vertices = []

        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    vertices.append(center + half_size * np.array([dx, dy, dz]))

        return np.array(vertices)

    def plot_metrics_comparison(self, metrics_by_algorithm: Dict[str, List[Dict]],
                               metric_name: str,
                               filename: str = None):
        """Plot comparison of specific metric across algorithms."""
        if filename is None:
            filename = f"{metric_name}_comparison.png"

        data = []
        labels = []

        for algo_name, metrics_list in metrics_by_algorithm.items():
            values = [m.get(metric_name, 0) for m in metrics_list]
            data.append(values)
            labels.append(algo_name)

        fig, ax = plt.subplots(figsize=(12, 6))

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color='red', linewidth=2),
                        boxprops=dict(facecolor='lightblue', alpha=0.7))

        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, data: np.ndarray,
                    x_labels: List[str],
                    y_labels: List[str],
                    title: str = "Heatmap",
                    filename: str = "heatmap.png",
                    cell_labels: Optional[List[List[str]]] = None):
        """Plot heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

        # Add values
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                label = cell_labels[i][j] if cell_labels is not None else f'{data[i, j]:.2f}'
                ax.text(j, i, label, ha="center", va="center", color="black", fontsize=8)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_statistical_comparison(self, statistical_results: Dict, filename: str = "statistical.png"):
        """Plot statistical test results."""
        if 'p_values' in statistical_results:
            p_values = np.array(statistical_results['p_values'])
            algorithms = statistical_results.get('algorithms', [])
            cell_labels = [
                [self._format_p_value(p_values[i, j], diagonal=(i == j))
                 for j in range(len(algorithms))]
                for i in range(len(algorithms))
            ]

            self.plot_heatmap(
                p_values,
                algorithms,
                algorithms,
                title=f"{statistical_results.get('method', 'Statistical Test')} - P-values",
                filename=filename,
                cell_labels=cell_labels
            )
            self._save_statistical_table(algorithms, cell_labels, filename)

    def _format_p_value(self, value: float, diagonal: bool = False) -> str:
        """Format p-values without rounding away very small values."""
        if diagonal:
            return "-"

        if not np.isfinite(value):
            return "NA"

        return f"{float(value):.17g}"

    def _save_statistical_table(self,
                                algorithms: List[str],
                                cell_labels: List[List[str]],
                                filename: str):
        """Save the exact p-value table next to the heatmap image."""
        output_path = self.results_dir / filename.replace(".png", ".csv")

        with output_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["algorithm", *algorithms])

            for algorithm, row in zip(algorithms, cell_labels):
                writer.writerow([algorithm, *row])

    def plot_pareto_fronts(self, pareto_fronts: Dict[str, tuple],
                          objective_names: List[str] = None,
                          filename: str = "pareto_fronts.png"):
        """
        Plot Pareto fronts for multi-objective algorithms.

        Args:
            pareto_fronts: Dict mapping algorithm name to (positions, objectives) tuple
            objective_names: Names of objectives to plot
            filename: Output filename
        """
        if not pareto_fronts:
            return

        # Get objectives from first algorithm
        sample_objectives = list(pareto_fronts.values())[0][1]
        if not sample_objectives:
            return

        obj_keys = list(sample_objectives[0].keys())
        if objective_names is None:
            objective_names = obj_keys

        # Handle 2D and 3D cases
        if len(obj_keys) == 2:
            self._plot_pareto_2d(pareto_fronts, obj_keys, filename)
        elif len(obj_keys) >= 3:
            # Plot first 3 objectives
            self._plot_pareto_3d(pareto_fronts, obj_keys[:3], filename)
            # Also create 2D projection plots
            self._plot_pareto_2d_projections(pareto_fronts, obj_keys, filename)
        else:
            # Single objective - just plot as 1D
            self._plot_pareto_1d(pareto_fronts, obj_keys[0], filename)

    def _plot_pareto_2d(self, pareto_fronts: Dict, obj_keys: List[str], filename: str):
        """Plot 2D Pareto fronts."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for algo_name, (positions, objectives) in pareto_fronts.items():
            if objectives:
                obj1_vals = [obj[obj_keys[0]] for obj in objectives]
                obj2_vals = [obj[obj_keys[1]] for obj in objectives]

                ax.scatter(obj1_vals, obj2_vals, label=algo_name.upper(),
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(obj_keys[0].replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(obj_keys[1].replace('_', ' ').title(), fontsize=12)
        ax.set_title('Pareto Fronts Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pareto_3d(self, pareto_fronts: Dict, obj_keys: List[str], filename: str):
        """Plot 3D Pareto fronts."""
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for algo_name, (positions, objectives) in pareto_fronts.items():
            if objectives:
                obj1_vals = [obj[obj_keys[0]] for obj in objectives]
                obj2_vals = [obj[obj_keys[1]] for obj in objectives]
                obj3_vals = [obj[obj_keys[2]] for obj in objectives]

                ax.scatter(obj1_vals, obj2_vals, obj3_vals, label=algo_name.upper(),
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel(obj_keys[0].replace('_', ' ').title(), fontsize=10)
        ax.set_ylabel(obj_keys[1].replace('_', ' ').title(), fontsize=10)
        ax.set_zlabel(obj_keys[2].replace('_', ' ').title(), fontsize=10)
        ax.set_title('3D Pareto Fronts Comparison', fontsize=14, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.results_dir / filename.replace('.png', '_3d.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pareto_2d_projections(self, pareto_fronts: Dict, obj_keys: List[str], filename: str):
        """Plot 2D projections of multi-objective Pareto fronts."""
        # Create pairwise objective plots
        n_objectives = len(obj_keys)
        if n_objectives < 2:
            return

        # Plot first 3 pairwise combinations
        pairs = [(0, 1), (0, 2), (1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (i, j) in enumerate(pairs):
            if i >= n_objectives or j >= n_objectives:
                continue

            ax = axes[idx]
            for algo_name, (positions, objectives) in pareto_fronts.items():
                if objectives:
                    obj_i_vals = [obj[obj_keys[i]] for obj in objectives]
                    obj_j_vals = [obj[obj_keys[j]] for obj in objectives]

                    ax.scatter(obj_i_vals, obj_j_vals, label=algo_name.upper(),
                              s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

            ax.set_xlabel(obj_keys[i].replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel(obj_keys[j].replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{obj_keys[i]} vs {obj_keys[j]}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle('Pareto Front Projections', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / filename.replace('.png', '_projections.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pareto_1d(self, pareto_fronts: Dict, obj_key: str, filename: str):
        """Plot 1D Pareto front (single objective)."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for algo_name, (positions, objectives) in pareto_fronts.items():
            if objectives:
                obj_vals = [obj[obj_key] for obj in objectives]
                x_vals = range(len(obj_vals))

                ax.scatter(x_vals, obj_vals, label=algo_name.upper(),
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Solution Index', fontsize=12)
        ax.set_ylabel(obj_key.replace('_', ' ').title(), fontsize=12)
        ax.set_title('Pareto Front Solutions', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
