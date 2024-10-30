import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':

    segments = [59.0,
                56.0,
                20.0,
                48.0,
                52.0,
                16.0,
                44.0,
                12.0,
                36.0,
                4.0,
                7.0,
                39.0,
                15.0,
                47.0,
                19.0,
                55.0,
                51.0,
                27.0,
                63.0,
                60.0,
                24.0,
                10.0,
                42.0,
                41.0,
                9.0,
                23.0]

    for i in range(0,len(segments)):
        # Load DataFrame
        df = pd.read_csv('agv.pkl', low_memory=False)

        # Select relevant columns and handle missing/invalid values
        df = df[['Timestamp', 'Battery cell voltage', 'X-coordinate', 'Y-coordinate', 'Heading', 'Going to ID', 'Current segment']]
        df['X-coordinate'] = pd.to_numeric(df['X-coordinate'], errors='coerce')
        df['Y-coordinate'] = pd.to_numeric(df['Y-coordinate'], errors='coerce')
        df['Battery cell voltage'] = pd.to_numeric(df['Battery cell voltage'], errors='coerce')
        df['Going to ID'] = pd.to_numeric(df['Going to ID'], errors='coerce')
        df['Heading'] = pd.to_numeric(df['Heading'], errors='coerce')
        df['Current segment'] = pd.to_numeric(df['Current segment'], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.index = df.pop('Timestamp')

        X = segments[i]

        # Filter data to segment X
        df = df[df['Current segment'] == X]

        distance_threshold = 1

        # Calculate Euclidean distance explicitly between each pair of consecutive points
        df['Distance'] = np.sqrt(
            (df['X-coordinate'] - df['X-coordinate'].shift(1))**2 + 
            (df['Y-coordinate'] - df['Y-coordinate'].shift(1))**2
        )

        # Initialize list for segmented dataframes
        dataframes = []
        start_idx = 0

        # Loop to split the dataframe based on the distance threshold
        for idx in range(1, len(df)):
            if df['Distance'].iloc[idx] > distance_threshold:
                segment_df = df.iloc[start_idx:idx].copy()  # Create a new segment
                dataframes.append(segment_df)               # Append segment to list
                start_idx = idx                             # Update start index for the next segment

        # Append the last segment if any rows remain
        if start_idx < len(df):
            segment_df = df.iloc[start_idx:].copy()
            dataframes.append(segment_df)

        # Optionally remove the Distance column from each segment if no longer needed
        for segment in dataframes:
            segment.drop(columns=['Distance'], inplace=True)

        # Number of points to resample (you can adjust this number)
        num_points = 100

        # Step 1: Interpolate each dataframe to have the same number of points
        interpolated_dfs = []
        for df in dataframes:
            # Create a fixed range of values from 0 to 1 to act as a new "index"
            resample_index = np.linspace(0, 1, num_points)
            # Interpolate X and Y coordinates using this new index
            df_interpolated = pd.DataFrame({
                'X-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['X-coordinate']),
                'Y-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['Y-coordinate'])
            })
            interpolated_dfs.append(df_interpolated)

        # Step 2: Concatenate and calculate the average for each coordinate across all dataframes
        average_df = pd.concat(interpolated_dfs).groupby(level=0).mean()

        # Step 3: Plot the averaged X and Y coordinates
        plt.figure(figsize=(10, 6))
        plt.plot(average_df['X-coordinate'], average_df['Y-coordinate'], label="Average Path", color="blue")

        # Add labels and title
        plt.title("Average Path of Segmented Dataframes")
        plt.xlabel("Average X-coordinate")
        plt.ylabel("Average Y-coordinate")
        plt.xlim(25, 60)  # Set x-axis limits
        plt.ylim(14, 28)  # Set y-axis limits
        plt.legend()
        plt.show()

        # Plot each segment as a separate line
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

        for i, segment_df in enumerate(dataframes):
            plt.plot(segment_df['X-coordinate'], segment_df['Y-coordinate'], label=f'Run {i+1}')

        # Add titles and labels
        plt.xlim(25, 60)  # Set x-axis limits
        plt.ylim(14, 28)  # Set y-axis limits
        plt.title("Plot of Segmented Dataframes for segment "+ str(X))
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.legend()  # Display a legend for each segment
        plt.show()

        # `dataframes` now contains all segmented dataframes