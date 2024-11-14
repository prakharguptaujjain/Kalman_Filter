import numpy as np

# Set parameters
time_steps = 50  # Total steps for tracking
delta_t = 0.001  # Time gap between state estimates (1 ms)
T = 0.01         # Time gap between successive RADAR readings (10 ms)
eta = int(T / delta_t)  # Number of state updates between RADAR readings

# Noise parameters
measurement_noise_cov = np.diag([0.1, 0.1, 0.1, 0.05, 0.05, 0.05])  # RADAR measurement noise
process_noise_cov = np.diag([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])  # Process noise

# State transition matrix
A = np.array([
    [1, 0, 0, delta_t, 0, 0],
    [0, 1, 0, 0, delta_t, 0],
    [0, 0, 1, 0, 0, delta_t],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# Control matrix
B = np.array([
    [0.5 * delta_t**2, 0, 0],
    [0, 0.5 * delta_t**2, 0],
    [0, 0, 0.5 * delta_t**2],
    [delta_t, 0, 0],
    [0, delta_t, 0],
    [0, 0, delta_t]
])

# Measurement matrix
H = np.eye(6)

# Initialize state and covariance
initial_position = np.array([0, 0, 0])  # Example initial position
initial_velocity = np.array([10, 10, 10])  # Example initial velocity
x_true = np.hstack([initial_position, initial_velocity]).astype(np.float64)  # True state vector
x_estimated = np.copy(x_true)  # Estimated state vector
P = np.eye(6)  # Initial estimation covariance

# Control (acceleration)
constant_acceleration = np.array([0.1, 0.1, -0.01])  # Example constant acceleration

# Tracking loop
errors = []
true_positions = []
estimated_positions = []

for step in range(time_steps):
    for i in range(eta):
        # True state propagation (without noise)
        x_true[3:] += constant_acceleration * delta_t  # Update velocity
        x_true[:3] += x_true[3:] * delta_t  # Update position

        # Kalman filter prediction step
        x_estimated = A @ x_estimated + B @ constant_acceleration
        P = A @ P @ A.T + process_noise_cov

    # Store true and estimated positions for plotting
    estimated_positions.append(x_estimated[:3].copy())

    # Generate noisy measurement from true state
    measurement_noise = np.random.multivariate_normal(np.zeros(6), measurement_noise_cov)
    measurement = H @ x_true + measurement_noise
    true_positions.append(measurement[:3].copy())

    # Kalman filter update step
    y = measurement - H @ x_estimated  # Measurement residual
    S = H @ P @ H.T + measurement_noise_cov  # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
    x_estimated += K @ y  # Updated state estimate
    P = (np.eye(6) - K @ H) @ P  # Updated covariance estimate

    # Calculate error (Euclidean distance) between true and estimated positions
    error = np.linalg.norm(x_true[:3] - x_estimated[:3])
    errors.append(error)

    print(f"Step {step+1}, Error: {error}")

# Convert position lists to numpy arrays for easier plotting
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Plotting the results
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(14, 6))

# Plot tracking error over time
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(errors, label="Tracking Error (Euclidean Distance)")
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Position Error")
ax1.set_title("Kalman Filter Tracking Error Over Time")
ax1.legend()

# 3D Plot of true vs estimated positions
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot(true_positions[:, 0], true_positions[:, 1], true_positions[:, 2], label="True Position")
ax2.plot(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], label="Estimated Position", linestyle='--')
ax2.set_xlabel("X Position")
ax2.set_ylabel("Y Position")
ax2.set_zlabel("Z Position")
ax2.set_title("True vs Estimated 3D Trajectories")
ax2.legend()

plt.tight_layout()
plt.show()
