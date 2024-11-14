import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


class kalmanFilterAbstract(ABC):
    """
    Abstract base class for a Kalman filter.

    Attributes:
        H (np.ndarray): Measurement matrix that relates the state to the measurements.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        x_estimated (np.ndarray): Current estimated state.
        P (np.ndarray): Current estimated error covariance.

    Methods:
        predict(x, u):
            Predicts the next state based on the current state and control input.
        update(x_pred, z):
            Updates the state estimate based on the predicted state and actual measurement.
    """

    def __init__(self, H, Q, R, initial_state, initial_covariance):
        """
        Initializes the Kalman filter with the measurement matrix, noise covariances, 
        initial state, and initial covariance.

        Args:
            H (np.ndarray): Measurement matrix.
            Q (np.ndarray): Process noise covariance.
            R (np.ndarray): Measurement noise covariance.
            initial_state (np.ndarray): Initial state estimate.
            initial_covariance (np.ndarray): Initial covariance estimate.
        """
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x_estimated = initial_state  # Initial state estimate
        self.P = initial_covariance  # Initial covariance estimate

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predicts the next state given the current state and control input.

        Args:
            x (np.ndarray): Current state vector.
            u (np.ndarray): Control input vector.z

        Returns:
            np.ndarray: Predicted state vector.
        """
        pass

    def update(self, x_pred: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Updates the state estimate based on the predicted state and actual measurement.

        Args:
            x_pred (np.ndarray): Predicted state vector.
            z (np.ndarray): Measurement vector.

        Returns:
            np.ndarray: Updated state vector.
        """
        pass
