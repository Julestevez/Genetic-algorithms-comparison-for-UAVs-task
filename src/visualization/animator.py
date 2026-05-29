"""Animation utilities for UAV swarm visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List
from pathlib import Path


class Animator:
    """Create animations of UAV swarm behavior."""

    def __init__(self, results_dir: str = "results/videos"):
        """Initialize animator."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def animate_swarm_3d(self,
                        uav_trajectories: Dict[int, np.ndarray],
                        obstacles: List[Dict] = None,
                        filename: str = "swarm_animation.mp4",
                        fps: int = 30):
        """
        Create 3D animation of swarm behavior.

        Args:
            uav_trajectories: Dict mapping UAV ID to trajectory array
            obstacles: List of obstacle dictionaries
            filename: Output filename
            fps: Frames per second
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get bounds
        all_positions = np.vstack([traj for traj in uav_trajectories.values()])
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

        padding = 10
        ax.set_xlim([x_min - padding, x_max + padding])
        ax.set_ylim([y_min - padding, y_max + padding])
        ax.set_zlim([max(0, z_min - padding), z_max + padding])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('UAV Swarm Animation')

        # Plot obstacles
        if obstacles:
            for obs in obstacles:
                self._plot_obstacle(ax, obs)

        # Initialize UAV plots with drone symbols
        uav_plots = {}
        trail_plots = {}

        # Create custom drone marker
        from matplotlib.path import Path
        import matplotlib.patches as mpatches

        # Drone shape: quadcopter with 4 arms
        drone_verts = [
            (0, 0),      # Center
            (-0.3, 0.3), # Top-left rotor
            (0, 0),      # Center
            (0.3, 0.3),  # Top-right rotor
            (0, 0),      # Center
            (-0.3, -0.3),# Bottom-left rotor
            (0, 0),      # Center
            (0.3, -0.3), # Bottom-right rotor
        ]

        for uav_id in uav_trajectories.keys():
            # Use drone marker (star marker as approximation for quadcopter)
            uav_plots[uav_id], = ax.plot([], [], [], marker='*', markersize=15,
                                        markeredgewidth=1.5, markeredgecolor='black',
                                        label=f'UAV {uav_id}', linestyle='none')
            trail_plots[uav_id], = ax.plot([], [], [], '-', alpha=0.3, linewidth=1)

        ax.legend(loc='upper right')

        max_frames = max(len(traj) for traj in uav_trajectories.values())

        def update(frame):
            for uav_id, traj in uav_trajectories.items():
                if frame < len(traj):
                    # Update UAV position
                    pos = traj[frame]
                    uav_plots[uav_id].set_data([pos[0]], [pos[1]])
                    uav_plots[uav_id].set_3d_properties([pos[2]])

                    # Update trail
                    trail_start = max(0, frame - 50)
                    trail = traj[trail_start:frame+1]
                    trail_plots[uav_id].set_data(trail[:, 0], trail[:, 1])
                    trail_plots[uav_id].set_3d_properties(trail[:, 2])

            return list(uav_plots.values()) + list(trail_plots.values())

        anim = FuncAnimation(fig, update, frames=max_frames, interval=1000/fps, blit=False)

        # Save animation
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        try:
            anim.save(str(self.results_dir / filename), writer=writer)
            print(f"Animation saved to: {self.results_dir / filename}")
        except Exception as e:
            print(f"Warning: Could not save animation. Error: {e}")
            print("Make sure ffmpeg is installed: brew install ffmpeg")

        plt.close()

    def _plot_obstacle(self, ax, obstacle: Dict):
        """Plot a single obstacle."""
        pos = obstacle['position']
        size = obstacle['size']

        # Create wireframe box
        r = size / 2
        vertices = [
            [pos[0] - r[0], pos[1] - r[1], pos[2] - r[2]],
            [pos[0] + r[0], pos[1] - r[1], pos[2] - r[2]],
            [pos[0] + r[0], pos[1] + r[1], pos[2] - r[2]],
            [pos[0] - r[0], pos[1] + r[1], pos[2] - r[2]],
            [pos[0] - r[0], pos[1] - r[1], pos[2] + r[2]],
            [pos[0] + r[0], pos[1] - r[1], pos[2] + r[2]],
            [pos[0] + r[0], pos[1] + r[1], pos[2] + r[2]],
            [pos[0] - r[0], pos[1] + r[1], pos[2] + r[2]],
        ]

        vertices = np.array(vertices)

        # Plot edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
        ]

        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'r-', alpha=0.6, linewidth=2)
