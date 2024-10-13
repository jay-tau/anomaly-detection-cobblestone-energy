import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

np.random.seed(0)

# Stream parameters
WINDOW_SIZE = 50
ALPHA = 0.3  # For EWMA
Z_THRESHOLD = 2.4375  # Z-score threshold for anomaly detection
SPIKE_THRESHOLD = 0.98  # Probability threshold for introducing spikes

# Function to simulate data stream with periodic signal and noise
def data_stream():
    t = 0
    while True:
        seasonal = 10 * np.sin(0.1 * t)  # Periodic signal
        noise = np.random.normal(0, 1)   # Random noise
        spike = 0
        if np.random.rand() > SPIKE_THRESHOLD:      # Introduce random spikes
            spike = np.random.uniform(20, 30)
        yield seasonal + noise + spike
        t += 1

# Z-score-based anomaly detection
def detect_anomaly(window):
    mean = np.mean(window)
    std = np.std(window)
    z_scores = [(x - mean) / std if std > 0 else 0 for x in window]
    return z_scores[-1] > Z_THRESHOLD or z_scores[-1] < -Z_THRESHOLD

# Exponentially Weighted Moving Average (EWMA) for gradual shifts
def ewma(prev_ewma, new_value):
    return ALPHA * new_value + (1 - ALPHA) * prev_ewma

# Initialize variables
data = deque(maxlen=WINDOW_SIZE)
moving_avg_data = deque(maxlen=WINDOW_SIZE)
rate_of_change_data = deque(maxlen=WINDOW_SIZE)
max_value, min_value = -np.inf, np.inf
ewma_value = 0

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Two subplots: one for raw/moving avg and one for rate of change
line, = ax1.plot([], [], lw=2, label="Raw Data")
moving_avg_line, = ax1.plot([], [], lw=2, label="Moving Average", color='orange')
anomalies, = ax1.plot([], [], 'ro', label='Anomalies')  # For marking anomalies
roc_line, = ax2.plot([], [], lw=2, label="Rate of Change (Moving Avg)", color='green')

# Initialize plot function
def init():
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-20, 30)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(-5, 5)
    
    # Add legends to both plots
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    
    return line, moving_avg_line, anomalies, roc_line

# Update function for animation
def update(frame):
    global max_value, min_value, ewma_value

    # Get new data point from stream
    new_point = next(stream)
    data.append(new_point)

    # Update max/min tracking
    if new_point > max_value:
        max_value = new_point
        print(f"New max value detected: {max_value}")
    if new_point < min_value:
        min_value = new_point
        print(f"New min value detected: {min_value}")

    # Update EWMA (Moving Average)
    ewma_value = ewma(ewma_value, new_point)
    moving_avg_data.append(ewma_value)

    # Compute rate of change for moving average
    if len(moving_avg_data) > 1:
        roc_value = moving_avg_data[-1] - moving_avg_data[-2]  # Rate of change (difference between consecutive values)
        rate_of_change_data.append(roc_value)
    else:
        rate_of_change_data.append(0)

    # Detect anomaly
    anomaly_flag = detect_anomaly(data)

    # Update plot data for raw data, moving average, and rate of change
    line.set_data(range(len(data)), list(data))
    moving_avg_line.set_data(range(len(moving_avg_data)), list(moving_avg_data))
    roc_line.set_data(range(len(rate_of_change_data)), list(rate_of_change_data))

    # Highlight anomaly if detected
    anomalies.set_data(range(len(data)), [new_point if anomaly_flag else np.nan for _ in data])

    return line, moving_avg_line, anomalies, roc_line

# Create the data stream generator
stream = data_stream()

# Run animation
ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True)
plt.show()
