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

    # Load DataFrame
    df = pd.read_csv('agv.pkl', low_memory=False)

    df = df[['Timestamp', 'Battery cell voltage', 'X-coordinate', 'Y-coordinate', 'Heading', 'Going to ID']]

    # Handle missing or invalid values
    df['X-coordinate'] = pd.to_numeric(df['X-coordinate'], errors='coerce')
    df['Y-coordinate'] = pd.to_numeric(df['Y-coordinate'], errors='coerce')
    df['Battery cell voltage'] = pd.to_numeric(df['Battery cell voltage'], errors='coerce')
    df['Going to ID'] = pd.to_numeric(df['Going to ID'], errors='coerce')
    df['Heading'] = pd.to_numeric(df['Heading'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Convert Timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.index = df.pop('Timestamp')

    # Normalize the data for features (first scaler)
    scaler_features = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[['Battery cell voltage', 'X-coordinate', 'Y-coordinate', 'Going to ID', 'Heading']] = scaler_features.fit_transform(
        df[['Battery cell voltage', 'X-coordinate', 'Y-coordinate', 'Going to ID', 'Heading']]
    )

    # Normalize the target data (second scaler)
    scaler_target = MinMaxScaler()
    df_target_scaled = scaler_target.fit_transform(df[['Battery cell voltage', 'X-coordinate', 'Y-coordinate']])

    # Create sequences (using df_scaled for features and df_target_scaled for targets)
    def create_sequences(df_features, df_target, n_steps):
        X, y = [], []
        for i in range(len(df_features) - n_steps):
            X.append(df_features.iloc[i:i+n_steps].values)
            y.append(df_target[i+n_steps])
        return np.array(X), np.array(y)

    n_steps = 3
    X, y = create_sequences(df_scaled[['Battery cell voltage', 'X-coordinate', 'Y-coordinate', 'Going to ID', 'Heading']],
                            df_target_scaled, n_steps)

    # Splitting data into training, validation, and test sets
    q_80 = int(len(X) * 0.8)
    q_90 = int(len(X) * 0.9)

    dates = df.index[3:]

    X_train, X_val, X_test = X[:q_80], X[q_80:q_90], X[q_90:]
    y_train, y_val, y_test = y[:q_80], y[q_80:q_90], y[q_90:]
    dates_train, dates_val, dates_test = dates[:q_80], dates[q_80:q_90], dates[q_90:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_steps, 5)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3))  # Output layer with 3 features (Battery cell voltage, X-coordinate, Y-coordinate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)  # Add gradient clipping
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_val, y_val), verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Inverse scaling for both y_test and y_pred to bring them back to the original scale
    y_test_rescaled = scaler_target.inverse_transform(y_test)
    y_pred_rescaled = scaler_target.inverse_transform(y_pred)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot real and predicted Battery Cell Voltage (first column in y)
    # plt.plot(dates_test, y_test_rescaled[:, 0], label='Real Battery Cell Voltage', color='blue', linewidth=2)
    # plt.plot(dates_test, y_pred_rescaled[:, 0], label='Predicted Battery Cell Voltage', color='blue', linestyle='dashed')

    # Plot real and predicted X-coordinate (second column in y)
    plt.plot(dates_test, y_test_rescaled[:, 1], label='Real X-coordinate', color='green', linewidth=2)
    plt.plot(dates_test, y_pred_rescaled[:, 1], label='Predicted X-coordinate', color='green', linestyle='dashed')

    # Plot real and predicted Y-coordinate (third column in y)
    plt.plot(dates_test, y_test_rescaled[:, 2], label='Real Y-coordinate', color='red', linewidth=2)
    plt.plot(dates_test, y_pred_rescaled[:, 2], label='Predicted Y-coordinate', color='red', linestyle='dashed')

    # Add title and labels
    #plt.title('Real vs Predicted Battery Cell Voltage, X-coordinate, and Y-coordinate')
    plt.title('Real vs Predicted X-coordinate, and Y-coordinate')
    plt.xlabel('Timestamp')
    plt.ylabel('Values')

    # Add a legend to differentiate real and predicted values
    plt.legend()

    # Show the plot
    plt.show()