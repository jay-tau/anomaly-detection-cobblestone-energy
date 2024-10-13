# Real-Time Anomaly Detection and Visualization

## Project Overview

This project implements real-time anomaly detection in a simulated data stream. The goal is to detect both abrupt and gradual shifts in the data by applying two techniques:

1. **Z-score based anomaly detection**: Identifies extreme outliers using the standard deviation from the data's mean.
2. **Exponentially Weighted Moving Average (EWMA)**: Tracks gradual shifts in the data and visualizes the rate of change.

The project also tracks new maximum and minimum values in the data stream and visualizes the raw data, moving average, and rate of change in real-time using `matplotlib`.

## Key Features

- **Real-Time Data Stream**: Simulates a continuous data stream with periodic signals and random noise.
- **Anomaly Detection**: Flags anomalies using the Z-score method, detecting data points that deviate significantly from the mean.
- **Moving Average**: Uses EWMA to smooth the data and monitor gradual shifts.
- **Rate of Change**: Calculates and plots the rate of change in the moving average.
- **Live Visualization**: Displays raw data, moving average, rate of change, and anomalies in real-time.

## How It Works

1. **Data Stream Simulation**: A function (`data_stream()`) generates a continuous stream of data that includes a seasonal pattern (sine wave) with added random noise and occasional spikes.
2. **Anomaly Detection**: The `detect_anomaly()` function computes the Z-score of the latest data point based on a sliding window of past values. If the Z-score exceeds a predefined threshold, the point is flagged as an anomaly.
3. **Moving Average**: The `ewma()` function calculates the exponentially weighted moving average to track gradual shifts in the data.
4. **Rate of Change**: The difference between consecutive moving average values is computed and plotted in a separate subplot to highlight changing trends.
5. **Visualization**: Using `matplotlib.animation`, the plot is updated in real-time, showing the raw data, moving average, anomalies, and rate of change.

## Installation

1. Clone this repository:

   ```bash
   git clone git@github.com:jay-tau/anomaly-detection-cobblestone-energy.git
   cd anomaly-detection-cobblestone-energy/
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *Note: The dependencies include `numpy`, `matplotlib`, and `scipy` for statistical functions and plotting.*

## Usage

Run the `main.py` script to start the real-time anomaly detection:

```bash
python main.py
```

The script will generate a plot with two subplots:

1. **Top Plot**: Displays the raw data, moving average, and detected anomalies.
2. **Bottom Plot**: Displays the rate of change in the moving average.

Anomalies are marked with red dots, and messages will be printed in the console whenever a new maximum or minimum value is detected.

## Parameters

- **WINDOW_SIZE**: The size of the sliding window for calculating statistics (default is 50).
- **ALPHA**: The smoothing factor for EWMA (default is 0.3).
- **Z_THRESHOLD**: The Z-score threshold for anomaly detection (default is 2.5).
- **SPIKE_THRESHOLD**: The probability threshold for random spikes in the data stream (default is 0.8).

## Example Output

- The real-time plot will update with new data points, showing anomalies as red points.
- The console will print new maximum and minimum values when detected.

## License

This project is licensed under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/).