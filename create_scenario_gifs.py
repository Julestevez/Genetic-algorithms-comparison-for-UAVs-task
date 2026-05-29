#!/usr/bin/env python3
"""
Enhanced GIF Generator for UAV Swarm - Creates Per-Scenario Visualizations
Shows all algorithms' UAV swarms in each scenario
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent / "src"))


class ScenarioGIFGenerator:
    """Generate GIFs showing all algorithms per scenario."""

    def __init__(self, results_file: str):
        """Initialize generator."""
        self.results_file = Path(results_file)

        with open(self.results_file) as f:
            self.results = json.load(f)

        # Create output directories
        timestamp = Path(results_file).stem.split('_')[-1]
        self.output_dir = Path(f"results/scenario_gifs_{timestamp}")
        self.gif_dir = self.output_dir / "per_scenario"
        self.combined_dir = self.output_dir / "combined"
        self.screenshot_dir = self.output_dir / "screenshots"

        for d in [self.gif_dir, self.combined_dir, self.screenshot_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"\nOutput directory: {self.output_dir}")

        # Organize results
        self.by_scenario = {}
        self.by_algorithm = {}

        for result in self.results:
            scenario = result['scenario']
            algorithm = result['algorithm']

            if scenario not in self.by_scenario:
                self.by_scenario[scenario] = {}
            if algorithm not in self.by_scenario[scenario]:
                self.by_scenario[scenario][algorithm] = []

            self.by_scenario[scenario][algorithm].append(result)

            if algorithm not in self.by_algorithm:
                self.by_algorithm[algorithm] = []
            self.by_algorithm[algorithm].append(result)

        print(f"Found {len(self.by_scenario)} scenarios and {len(self.by_algorithm)} algorithms")

    def generate_3d_positions(self, best_position, num_uavs, num_frames=100):
        """Generate 3D trajectory animation frames from best position."""
        # best_position is flattened: [x1,y1,z1, x2,y2,z2, ...]
        best_position = np.array(best_position)

        # Reshape to (num_uavs, 3)
        try:
            positions = best_position.reshape(num_uavs, 3)
        except:
            # If shape doesn't match, use synthetic data
            positions = np.random.uniform([10, 10, 5], [90, 90, 35], (num_uavs, 3))

        # Generate animation: start at random positions, converge to best
        start_positions = np.random.uniform([0, 0, 0], [100, 100, 40], (num_uavs, 3))

        frames = []
        for i in range(num_frames):
            progress = i / num_frames
            # Smooth interpolation
            progress_smooth = 0.5 - 0.5 * np.cos(progress * np.pi)
            current = start_positions + (positions - start_positions) * progress_smooth
            frames.append(current)

        return frames

    def create_scenario_gif(self, scenario_name):
        """Create GIF for one scenario showing all algorithms."""
        print(f"\nCreating GIF for scenario: {scenario_name}")

        if scenario_name not in self.by_scenario:
            print(f"  No results for scenario: {scenario_name}")
            return

        algo_results = self.by_scenario[scenario_name]
        algorithms = list(algo_results.keys())

        if not algorithms:
            print(f"  No algorithms for scenario: {scenario_name}")
            return

        # Get first result from each algorithm
        algo_data = {}
        num_uavs = 5  # default

        for algo in algorithms:
            if algo_results[algo]:
                result = algo_results[algo][0]  # Take first trial
                num_uavs = len(result.get('best_position', [])) // 3
                if num_uavs == 0:
                    num_uavs = 5
                frames = self.generate_3d_positions(result['best_position'], num_uavs)
                algo_data[algo] = frames

        if not algo_data:
            print(f"  No valid data for scenario: {scenario_name}")
            return

        num_frames = len(frames)

        # Create figure with 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 40)
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title(f'{scenario_name.replace("_", " ").title()}\nAll Algorithms UAV Swarms',
                    fontsize=14, fontweight='bold')

        # Color map for algorithms
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}

        # Initialize scatter plots
        scatter_plots = {}
        trail_plots = {}

        for algo in algorithms:
            scatter_plots[algo] = ax.scatter([], [], [],
                                            c=algo_colors[algo],
                                            marker='o',
                                            s=300,
                                            alpha=0.9,
                                            label=algo)
            trail_plots[algo] = ax.plot([], [], [],
                                       color=algo_colors[algo],
                                       alpha=0.3,
                                       linewidth=1)[0]

        # Legend
        ax.legend(loc='upper right', fontsize=10)

        # Text annotation
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Store trails
        trails = {algo: {uav: [] for uav in range(num_uavs)} for algo in algorithms}

        def init():
            for algo in algorithms:
                scatter_plots[algo]._offsets3d = ([], [], [])
                trail_plots[algo].set_data([], [])
                trail_plots[algo].set_3d_properties([])
            time_text.set_text('')
            return list(scatter_plots.values()) + list(trail_plots.values()) + [time_text]

        def animate(frame):
            for algo in algorithms:
                if algo in algo_data:
                    positions = algo_data[algo][frame]

                    # Update scatter
                    scatter_plots[algo]._offsets3d = (
                        positions[:, 0],
                        positions[:, 1],
                        positions[:, 2]
                    )

                    # Update trails (show last 20 frames)
                    if frame > 0:
                        for uav_idx in range(len(positions)):
                            trails[algo][uav_idx].append(positions[uav_idx])
                            if len(trails[algo][uav_idx]) > 20:
                                trails[algo][uav_idx].pop(0)

            # Update time
            progress = frame / num_frames * 100
            time_text.set_text(f'Progress: {progress:.1f}%\nFrame: {frame}/{num_frames}')

            return list(scatter_plots.values()) + list(trail_plots.values()) + [time_text]

        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=num_frames, interval=50, blit=False)

        # Save GIF
        gif_path = self.gif_dir / f"{scenario_name}_all_algorithms.gif"
        print(f"  Saving GIF (this may take a minute)...")
        anim.save(str(gif_path), writer=PillowWriter(fps=20))
        plt.close()

        print(f"  ✓ Created: {gif_path.name}")

        # Save sample screenshots
        for frame_idx in [0, num_frames//2, num_frames-1]:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 40)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{scenario_name.replace("_", " ").title()} - Frame {frame_idx}')

            for algo in algorithms:
                if algo in algo_data:
                    positions = algo_data[algo][frame_idx]
                    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=algo_colors[algo], marker='o', s=300, alpha=0.9, label=algo)

            ax.legend()

            screenshot_path = self.screenshot_dir / f"{scenario_name}_frame_{frame_idx:03d}.png"
            plt.savefig(screenshot_path, dpi=150, bbox_inches='tight')
            plt.close()

        return gif_path

    def create_combined_algorithms_gif(self):
        """Create one GIF showing all algorithms across scenarios."""
        print("\nCreating combined algorithms comparison GIF...")

        # Pick one representative scenario for each algorithm
        representative_scenario = list(self.by_scenario.keys())[0]

        if representative_scenario not in self.by_scenario:
            print("  No scenarios available")
            return

        algo_results = self.by_scenario[representative_scenario]
        algorithms = list(algo_results.keys())

        if len(algorithms) < 2:
            print("  Need at least 2 algorithms")
            return

        # Create subplots for each algorithm
        num_algos = len(algorithms)
        cols = min(3, num_algos)
        rows = (num_algos + cols - 1) // cols

        fig = plt.figure(figsize=(6*cols, 5*rows))
        fig.suptitle('Algorithm Comparison - UAV Swarm Trajectories',
                    fontsize=16, fontweight='bold')

        axes = []
        scatter_plots = []
        algo_frames = []

        for idx, algo in enumerate(algorithms, 1):
            ax = fig.add_subplot(rows, cols, idx, projection='3d')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 40)
            ax.set_xlabel('X (m)', fontsize=9)
            ax.set_ylabel('Y (m)', fontsize=9)
            ax.set_zlabel('Z (m)', fontsize=9)
            ax.set_title(algo, fontsize=12, fontweight='bold')
            axes.append(ax)

            # Generate frames for this algorithm
            if algo_results[algo]:
                result = algo_results[algo][0]
                num_uavs = len(result.get('best_position', [])) // 3
                if num_uavs == 0:
                    num_uavs = 5
                frames = self.generate_3d_positions(result['best_position'], num_uavs)
                algo_frames.append(frames)

                scatter = ax.scatter([], [], [], c='blue', marker='o', s=80, alpha=0.8)
                scatter_plots.append(scatter)
            else:
                algo_frames.append(None)
                scatter_plots.append(None)

        num_frames = len(algo_frames[0]) if algo_frames[0] is not None else 50

        def init():
            for scatter in scatter_plots:
                if scatter is not None:
                    scatter._offsets3d = ([], [], [])
            return scatter_plots

        def animate(frame):
            for idx, (scatter, frames) in enumerate(zip(scatter_plots, algo_frames)):
                if scatter is not None and frames is not None:
                    positions = frames[frame]
                    scatter._offsets3d = (
                        positions[:, 0],
                        positions[:, 1],
                        positions[:, 2]
                    )
            return scatter_plots

        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=num_frames, interval=50, blit=False)

        gif_path = self.combined_dir / "all_algorithms_comparison.gif"
        print(f"  Saving combined GIF...")
        anim.save(str(gif_path), writer=PillowWriter(fps=20))
        plt.close()

        print(f"  ✓ Created: {gif_path.name}")
        return gif_path

    def generate_all(self):
        """Generate all GIFs."""
        print("\n" + "="*80)
        print("SCENARIO GIF GENERATION")
        print("="*80)

        # Per-scenario GIFs
        print("\nCreating per-scenario GIFs...")
        scenario_gifs = []
        for scenario in self.by_scenario.keys():
            gif = self.create_scenario_gif(scenario)
            if gif:
                scenario_gifs.append(gif)

        # Combined GIF
        print("\nCreating combined algorithms GIF...")
        combined_gif = self.create_combined_algorithms_gif()

        # Summary
        print("\n" + "="*80)
        print("GIF GENERATION COMPLETE!")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"  Per-scenario GIFs: {self.gif_dir}")
        print(f"  Combined GIF: {self.combined_dir}")
        print(f"  Screenshots: {self.screenshot_dir}")
        print(f"\nGenerated:")
        print(f"  {len(scenario_gifs)} scenario GIFs")
        print(f"  1 combined algorithm comparison GIF")
        print(f"  {len(list(self.screenshot_dir.glob('*.png')))} screenshots")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_scenario_gifs.py <benchmark_results.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    generator = ScenarioGIFGenerator(results_file)
    generator.generate_all()


if __name__ == '__main__':
    main()
