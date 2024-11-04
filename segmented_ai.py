import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_ai(df, segment_id):

    # Normalize the data for features (first scaler)
    scaler_features = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[['X-coordinate', 'Y-coordinate', 'Heading']] = scaler_features.fit_transform(
        df[['X-coordinate', 'Y-coordinate','Heading']]
    )

    # Normalize the target data (second scaler)
    scaler_target = MinMaxScaler()
    df_target_scaled = scaler_target.fit_transform(df[['X-coordinate', 'Y-coordinate','Heading']])

    # Create sequences (using df_scaled for features and df_target_scaled for targets)
    def create_sequences(df_features, df_target, n_steps):
        X, y = [], []
        for i in range(len(df_features) - n_steps):
            X.append(df_features.iloc[i:i+n_steps].values)
            y.append(df_target[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = 5
    X, y = create_sequences(df_scaled[[ 'X-coordinate', 'Y-coordinate','Heading'  ]],
                            df_target_scaled, n_steps)
    
    X_val = X[:int(len(X)/3)]
    y_val = y[:int(len(y)/3)]

    model = Sequential([
        LSTM(50, input_shape=(n_steps, 3)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model
    model.fit(X, y, epochs=300, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    model.save('keras/segment_'+str(segment_id)+'.keras')

def create_dataframes():
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
    
    # Load DataFrame
    full_df = pd.read_csv('agv.pkl', low_memory=False)

    # Select relevant columns and handle missing/invalid values
    full_df = full_df[['Timestamp', 'X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']]
    full_df['X-coordinate'] = pd.to_numeric(full_df['X-coordinate'], errors='coerce')
    full_df['Y-coordinate'] = pd.to_numeric(full_df['Y-coordinate'], errors='coerce')
    full_df['Heading'] = pd.to_numeric(full_df['Heading'], errors='coerce')
    full_df['Current segment'] = pd.to_numeric(full_df['Current segment'], errors='coerce')

    # Drop rows with NaN values
    full_df = full_df.dropna()

    # Set 'Timestamp' as index
    full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'])
    full_df.index = full_df.pop('Timestamp')

    for i in range(0,len(segments)):
        
        # Assign current segment from an array
        X = segments[i]

        # Copy original dataframe 
        df = full_df.copy()

        # Filter data to segment X
        df = df[df['Current segment'] == X]

        # Remove rows where Current segment is 56 and Y-coordinate > 18
        if X == 56.0:
            df = df[~(df['Y-coordinate'] > 18)]

        # Set distance for max distance between two consecutive points
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

        # Initialize variable for computing average dataframe length  

        avg_len = 0

        # Optionally remove the Distance column from each segment if no longer needed
        for segment in dataframes:
            segment.drop(columns=['Distance'], inplace=True)
        # Compute average amount of points in one run in segment 
            avg_len += len(segment)

        avg_len /= len(dataframes)


        # Number of points to resample is equal to average amount of points in one run in segment
        num_points = int(avg_len)

        # Interpolate each dataframe to have the same number of points
        interpolated_dfs = []
        for df in dataframes:
            # Create a fixed range of values from 0 to 1 to act as a new "index"
            resample_index = np.linspace(0, 1, num_points)
            # Interpolate heading, X and Y coordinates using this new index
            df_interpolated = pd.DataFrame({
                'X-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['X-coordinate']),
                'Y-coordinate': np.interp(resample_index, np.linspace(0, 1, len(df)), df['Y-coordinate']),
                'Heading': np.interp(resample_index, np.linspace(0, 1, len(df)), df['Heading'])
            })
            interpolated_dfs.append(df_interpolated)

        #  Concatenate and calculate the average for each coordinate across all dataframes
        average_df = pd.concat(interpolated_dfs).groupby(level=0).mean()
        create_ai(average_df, X)


if __name__ == '__main__':
    create_dataframes()