import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

import numpy as np
from kalmanFilter.kalmanFilter import kalmanFilterAbstract

class MissileTrackerKalmanFilter(kalmanFilterAbstract):
    """
    Kalman filter implementation for missile tracking with constant velocity and acceleration.

    Attributes:
        delta_t (float): Time step between predictions.
        A (np.ndarray): State transition matrix.
        B (np.ndarray): Control input matrix.

    Methods:
        predict(control_input):
            Predicts the next state based on control input.
        update(measurement):
            Updates the state estimate with the new measurement.
    """

    def __init__(self, delta_t, H, Q, R, initial_state, initial_covariance):
        """
        Initializes the MissileTrackerKalmanFilter with necessary parameters.

        Args:
            delta_t (float): Time step between predictions.
            H (np.ndarray): Measurement matrix.
            Q (np.ndarray): Process noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            initial_state (np.ndarray): Initial state estimate.
            initial_covariance (np.ndarray): Initial covariance estimate.
        """
        super().__init__(H, Q, R, initial_state, initial_covariance)
        self.delta_t = delta_t
        self.A = np.array([
            [1, 0, 0, delta_t, 0, 0],
            [0, 1, 0, 0, delta_t, 0],
            [0, 0, 1, 0, 0, delta_t],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.B = np.array([
            [0.5 * delta_t**2, 0, 0],
            [0, 0.5 * delta_t**2, 0],
            [0, 0, 0.5 * delta_t**2],
            [delta_t, 0, 0],
            [0, delta_t, 0],
            [0, 0, delta_t]
        ])

    def predict(self, control_input):
        """
        Performs the prediction step for the Kalman filter using the control input.

        Args:
            control_input (np.ndarray): Acceleration input vector.

        Returns:
            None
        """
        # Prediction step
        self.x_estimated = self.A @ self.x_estimated + self.B @ control_input
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measurement):
        """
        Performs the update step for the Kalman filter using the measurement.

        Args:
            measurement (np.ndarray): Observed measurement vector.

        Returns:
            None
        """
        # Measurement update step
        y = measurement - self.H @ self.x_estimated  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x_estimated += K @ y  # Updated state estimate
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P  # Updated covariance estimate