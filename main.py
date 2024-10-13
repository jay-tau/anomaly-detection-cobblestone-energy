import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# Stream parameters
WINDOW_SIZE = 50
ALPHA = 0.3  # For EWMA
threshold = 3  # Z-score threshold for anomaly detection

# Function to simulate data stream with periodic signal and noise
def data_stream():
    t = 0
    while True:
        seasonal = 10 * np.sin(0.1 * t)  # Periodic signal
        noise = np.random.normal(0, 1)   # Random noise
        spike = 0
        if np.random.rand() > 0.98:      # Introduce random spikes
            spike = np.random.uniform(10, 20)
        yield seasonal + noise + spike
        t += 1

# Z-score-based anomaly detection
def detect_anomaly(window):
    mean = np.mean(window)
    std = np.std(window)
    z_scores = [(x - mean) / std if std > 0 else 0 for x in window]
    return z_scores[-1] > threshold or z_scores[-1] < -threshold

# Exponentially Weighted Moving Average (EWMA) for gradual shifts
def ewma(prev_ewma, new_value):
    return ALPHA * new_value + (1 - ALPHA) * prev_ewma

# Initialize variables
data = deque(maxlen=WINDOW_SIZE)
max_value, min_value = -np.inf, np.inf
ewma_value = 0

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
anomalies, = ax.plot([], [], 'ro', label='Anomalies')  # For marking anomalies

# Initialize plot function
def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(-20, 30)
    return line, anomalies

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

    # Update EWMA
    ewma_value = ewma(ewma_value, new_point)

    # Detect anomaly
    anomaly_flag = detect_anomaly(data)

    # Update plot data
    line.set_data(range(len(data)), list(data))

    # Highlight anomaly if detected
    anomalies.set_data(range(len(data)), [new_point if anomaly_flag else np.nan for _ in data])

    return line, anomalies

# Create the data stream generator
stream = data_stream()

# Run animation
ani = FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True)
plt.legend()
plt.show()
