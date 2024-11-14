from utilsKalman.util import Util
import matplotlib.pyplot as plt
import numpy as np
from kalmanFilter.kalmanFilterImplementation import MissileTrackerKalmanFilter

import streamlit as st
import numpy as np
import os
import re

# Constant Acceleration
def experiment1(delta_t, time_steps, initial_position, initial_velocity, H, initial_state, initial_covariance,
                measurement_noise_multiplier=1, process_noise_multiplier=1, acceleration_multiplier=1, normal_run=False):
    """
    Simulates and plots a trajectory with constant acceleration using a Kalman filter.

    Args:
        delta_t (float): Time interval between steps.
        time_steps (int): Number of time steps to simulate.
        initial_position (np.ndarray): Initial position vector.
        initial_velocity (np.ndarray): Initial velocity vector.
        H (np.ndarray): Measurement matrix.
        initial_state (np.ndarray): Initial state estimate.
        initial_covariance (np.ndarray): Initial covariance estimate.
        measurement_noise_multiplier (float, optional): Multiplier for measurement noise. Defaults to 1.
        process_noise_multiplier (float, optional): Multiplier for process noise. Defaults to 1.
        acceleration_multiplier (float, optional): Multiplier for constant acceleration. Defaults to 1.
        normal_run (bool, optional): If True, shows the plot. Defaults to False.

    Returns:
        None
    """
    measurement_noise_cov = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])*measurement_noise_multiplier
    process_noise_cov = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01])*process_noise_multiplier
    constant_acceleration = np.array([0.1, 0.1, -0.01])*acceleration_multiplier
    T = 0.01
    eta = int(T / delta_t)

    kalman_filter = MissileTrackerKalmanFilter(delta_t, H, process_noise_cov, measurement_noise_cov, initial_state, initial_covariance)
    true_positions, estimated_positions, errors = Util.simulate_trajectory(
        time_steps, delta_t, eta, initial_position, initial_velocity, constant_acceleration, kalman_filter
    )
    fig=Util.plot_trajectory(true_positions, estimated_positions, errors)
    st.pyplot(fig)

# Variable Acceleration
def experiment2(delta_t, time_steps, initial_position, initial_velocity, H, initial_state, initial_covariance,
                measurement_noise_multiplier=1, process_noise_multiplier=1, acceleration_multiplier=1, normal_run=False):
    """
    Simulates and plots a trajectory with variable acceleration using a Kalman filter.

    Args:
        delta_t (float): Time interval between steps.
        time_steps (int): Number of time steps to simulate.
        initial_position (np.ndarray): Initial position vector.
        initial_velocity (np.ndarray): Initial velocity vector.
        H (np.ndarray): Measurement matrix.
        initial_state (np.ndarray): Initial state estimate.
        initial_covariance (np.ndarray): Initial covariance estimate.
        measurement_noise_multiplier (float, optional): Multiplier for measurement noise. Defaults to 1.
        process_noise_multiplier (float, optional): Multiplier for process noise. Defaults to 1.
        acceleration_multiplier (float, optional): Multiplier for variable acceleration. Defaults to 1.
        normal_run (bool, optional): If True, shows the plot. Defaults to False.

    Returns:
        None
    """
    measurement_noise_cov = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]) * measurement_noise_multiplier
    process_noise_cov = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]) * process_noise_multiplier
    variable_acceleration = lambda t: np.array([0.1 * np.sin(t), 0.1 * np.cos(t), -0.02 * t] * acceleration_multiplier)
    T = 0.02

    kalman_filter = MissileTrackerKalmanFilter(delta_t, H, process_noise_cov, measurement_noise_cov, initial_state, initial_covariance)
    true_positions, estimated_positions, errors = Util.simulate_trajectory(
        time_steps, delta_t, int(T / delta_t), initial_position, initial_velocity, variable_acceleration, kalman_filter
    )
    fig=Util.plot_trajectory(true_positions, estimated_positions, errors)
    st.pyplot(fig)

# Streamlit app
def streamlit():
    """
    Streamlit-based interface for running Kalman filter experiments with adjustable parameters.

    Allows the user to select between constant and variable acceleration experiments, adjusting 
    simulation parameters such as delta time, time steps, noise multipliers, and acceleration 
    multiplier via sliders.

    Returns:
        None
    """
    st.sidebar.title("Kalman Filter Experiments")
    page = st.sidebar.selectbox("Constant Acceleration", ["Constant Acceleration", "Variable Acceleration"])

    if page == "Constant Acceleration":
        delta_t = st.sidebar.slider("Delta Time", min_value=0.001, max_value=0.01, value=0.001, step=0.001)
    elif page == "Variable Acceleration":
        delta_t = st.sidebar.slider("Delta Time", min_value=0.001, max_value=0.02, value=0.001, step=0.001)
        
    time_steps = st.sidebar.slider("Time Steps", min_value=10, max_value=100, value=50, step=1)
    initial_position = np.array([0, 0, 0])
    initial_velocity = np.array([10, 10, 10])
    H = np.eye(6)
    initial_state = np.hstack([initial_position, initial_velocity])
    initial_covariance = np.eye(6)
    measurement_noise_multiplier = st.sidebar.slider("Measurement Noise Multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    process_noise_multiplier = st.sidebar.slider("Process Noise Multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    acceleration_multiplier = st.sidebar.slider("Acceleration Multiplier", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

    st.title(f"{page}")
    
    if page == "Constant Acceleration":
        experiment1(delta_t, time_steps, initial_position, initial_velocity, H, initial_state, initial_covariance,measurement_noise_multiplier, process_noise_multiplier, acceleration_multiplier)
    elif page == "Variable Acceleration":
        experiment2(delta_t, time_steps, initial_position, initial_velocity, H, initial_state, initial_covariance,measurement_noise_multiplier, process_noise_multiplier, acceleration_multiplier)


def normal_run():
    """
    Runs multiple Kalman filter experiments with different noise levels, acceleration types,
    and time intervals.

    Experiments include:
        - Constant acceleration with varying process and measurement noise.
        - Time-dependent acceleration with varying noise.
        - Different time step (delta_t) values to observe performance impact.

    Results are displayed for each configuration and can be saved for analysis.

    Returns:
        None
    """
    # Default initial conditions
    initial_position = np.array([0, 0, 0])
    initial_velocity = np.array([10, 10, 10])
    H = np.eye(6)
    initial_state = np.hstack([initial_position, initial_velocity])
    initial_covariance = np.eye(6)
    time_steps = 50

    # Experiment configurations
    delta_t_values = [0.001, 0.005, 0.01]  # Different time intervals
    noise_multipliers = [(1, 1), (0.5, 0.5), (2, 2)]  # Different levels of process and measurement noise
    acceleration_types = [
        ("Constant", np.array([0.1, 0.1, -0.01])),
        ("Time-dependent", lambda t: np.array([0.1 * np.sin(t), 0.1 * np.cos(t), -0.02 * t]))
    ]

    for delta_t in delta_t_values:
        for (proc_noise_mult, meas_noise_mult) in noise_multipliers:
            for acc_type, acceleration in acceleration_types:
                # Create Kalman filter with adjusted noise
                process_noise_cov = np.diag([0.02, 0.02, 0.02, 0.01, 0.01, 0.01]) * proc_noise_mult
                measurement_noise_cov = np.diag([0.05, 0.05, 0.05, 0.02, 0.02, 0.02]) * meas_noise_mult

                # Instantiate Kalman filter
                kalman_filter = MissileTrackerKalmanFilter(
                    delta_t, H, process_noise_cov, measurement_noise_cov, initial_state, initial_covariance
                )

                # Run simulation based on acceleration type
                if acc_type == "Constant":
                    true_positions, estimated_positions, errors = Util.simulate_trajectory(
                        time_steps, delta_t, int(0.01 / delta_t), initial_position, initial_velocity, acceleration, kalman_filter
                    )
                elif acc_type == "Time-dependent":
                    true_positions, estimated_positions, errors = Util.simulate_trajectory(
                        time_steps, delta_t, int(0.02 / delta_t), initial_position, initial_velocity, acceleration, kalman_filter
                    )

                # Plot and display results
                title = f"Experiment: Δt={delta_t}, Proc Noise Mult={proc_noise_mult}, Meas Noise Mult={meas_noise_mult}, Accel={acc_type}"
                print(title)
                fig = Util.plot_trajectory(true_positions, estimated_positions, errors, normal_run=False)
                # save fig
                save_name = re.sub(r'[<>:"/\\|?*]', "", title)
                save_name = save_name.replace(" ", "_").replace("Δ", "delta").replace("=", "_").replace(",", "_").replace(".", "_") + ".png"
                os.makedirs("results", exist_ok=True)
                fig.savefig(os.path.join("results", save_name))

if __name__ == '__main__':
    # streamlit()
    normal_run()
