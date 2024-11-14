import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from kalmanFilter.kalmanFilterImplementation import MissileTrackerKalmanFilter

class Util:
    """
    Utility class containing methods for simulating and plotting trajectories.

    Methods:
        simulate_trajectory(time_steps, delta_t, eta, initial_position, 
                            initial_velocity, acceleration, kalman_filter):
            Simulates a trajectory with true and estimated positions and computes errors.

        plot_trajectory(true_positions, estimated_positions, errors, normal_run=False):
            Plots the true vs. estimated trajectory and the tracking error over time.
    """
    @staticmethod
    def simulate_trajectory(time_steps, delta_t, eta, initial_position, initial_velocity, acceleration, kalman_filter):
        """
        Simulates the trajectory of a moving object using a Kalman filter.

        Args:
            time_steps (int): Number of time steps to simulate.
            delta_t (float): Time interval between steps.
            eta (int): Number of prediction steps per time step.
            initial_position (np.ndarray): Initial position vector.
            initial_velocity (np.ndarray): Initial velocity vector.
            acceleration (callable or np.ndarray): Constant or time-dependent acceleration.
            kalman_filter (kalmanFilterAbstract): An instance of a Kalman filter.

        Returns:
            np.ndarray: True positions of the object over time.
            np.ndarray: Estimated positions from the Kalman filter.
            list: Euclidean errors between true and estimated positions.
        """
        true_positions = []
        estimated_positions = []
        errors = []

        x_true = np.hstack([initial_position, initial_velocity]).astype(np.float64)

        for step in range(time_steps):
            for _ in range(eta):
                if callable(acceleration):
                    current_acceleration = acceleration(step * delta_t)
                else:
                    current_acceleration = acceleration

                # Update true state
                x_true[3:] += current_acceleration * delta_t
                x_true[:3] += x_true[3:] * delta_t

                # Kalman filter prediction step
                kalman_filter.predict(current_acceleration)

            measurement_noise = np.random.multivariate_normal(np.zeros(6), kalman_filter.R)
            measurement = kalman_filter.H @ x_true + measurement_noise

            kalman_filter.update(measurement)

            estimated_positions.append(kalman_filter.x_estimated[:3].copy())
            true_positions.append(measurement[:3].copy())
            error = np.linalg.norm(x_true[:3] - kalman_filter.x_estimated[:3])
            errors.append(error)

        return np.array(true_positions), np.array(estimated_positions), errors
    
    @staticmethod
    def plot_trajectory(true_positions, estimated_positions, errors, normal_run=False):
        """
        Plots the true and estimated positions in 3D and the position error over time.

        Args:
            true_positions (np.ndarray): True positions of the object.
            estimated_positions (np.ndarray): Positions estimated by the Kalman filter.
            errors (list): Position errors over time.
            normal_run (bool, optional): If True, displays the plot. Defaults to False.

        Returns:
            plt.Figure: The matplotlib figure object containing the plots.
        """
        fig = plt.figure(figsize=(14, 6))

        # 3D Trajectory plot
        ax2 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label="True Position")
        ax2.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label="Estimated Position", linestyle='--')
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        ax2.set_zlabel("Z Position")
        ax2.set_title("True vs Estimated 3D Trajectories")
        ax2.legend()

        # Error plot
        ax1 = fig.add_subplot(1, 2, 2)
        ax1.plot(errors, label="Tracking Error (Euclidean Distance)")
        ax1.legend(loc='upper left')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Position Error")
        ax1.set_title("Kalman Filter Tracking Error Over Time")
        ax1.legend()

        if normal_run:
            plt.show()
        return fig