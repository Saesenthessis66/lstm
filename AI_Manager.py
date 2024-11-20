import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class AI_Manager:

    _n_steps : int = 20
    _scaler = any

    def __init__(self, steps : int):
        self._n_steps = steps
        self._scaler = joblib.load('scaler.pkl')


    def createLSTM(self, data):
        features = ['X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']
        targets = ['X-coordinate', 'Y-coordinate', 'Heading']
        all_columns = features  # Features and targets overlap

        # Initialize a single scaler
        scaler = MinMaxScaler()

        # Normalize the data (combine features and targets for scaling)
        scaler.fit(data[1][all_columns])
        joblib.dump(scaler, 'scaler.pkl')

        data[0][all_columns] = scaler.transform(data[0][all_columns])
        data[1][all_columns] = scaler.transform(data[1][all_columns])

        # Create sequences
        sequence_length = 10  # Number of timesteps in each sequence
        X_train, X_val, y_train, y_val = [], [], [], []

        for i in range(len(data[1]) - sequence_length):
            X_train.append(data[1][features].iloc[i:i + sequence_length].values)
            y_train.append(data[1][targets].iloc[i + sequence_length].values)

        for i in range(len(data[0]) - sequence_length):
            X_val.append(data[0][features].iloc[i:i + sequence_length].values)
            y_val.append(data[0][targets].iloc[i + sequence_length].values)

        X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

        # Build the LSTM model
        model = Sequential([
            LSTM(64, input_shape=(sequence_length, len(features)), return_sequences=False),
            Dense(32, activation='relu'),
            Dense(3)  # Output layer for X, Y, Heading
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.summary()

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=2
        )

        model.save('model.keras')

        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=2)
        print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

        # Make predictions
        predictions = model.predict(X_val)

        # Add a dummy column for 'Current segment' since it was included in the scaler fitting
        dummy_column = np.zeros((y_val.shape[0], 1))  # Create a column of zeros
        predictions_with_dummy = np.hstack([predictions, dummy_column])  # Add dummy column to predictions
        y_val_with_dummy = np.hstack([y_val, dummy_column])  # Add dummy column to true values

        # Inverse transform using the scaler
        predictions_original = scaler.inverse_transform(predictions_with_dummy)[:, :3]  # Extract only the original 3 columns
        y_val_original = scaler.inverse_transform(y_val_with_dummy)[:, :3]  # Extract only the original 3 columns

        # Extract specific columns for individual plots
        true_x = y_val_original[:, 0]
        predicted_x = predictions_original[:, 0]

        true_y = y_val_original[:, 1]
        predicted_y = predictions_original[:, 1]

        true_heading = y_val_original[:, 2]
        predicted_heading = predictions_original[:, 2]

        # Plot X-coordinate
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(true_x, label='True X')
        plt.plot(predicted_x, label='Predicted X', linestyle='dashed')
        plt.title('X-coordinate')
        plt.legend()

        # Plot Y-coordinate
        plt.subplot(1, 3, 2)
        plt.plot(true_y, label='True Y')
        plt.plot(predicted_y, label='Predicted Y', linestyle='dashed')
        plt.title('Y-coordinate')
        plt.legend()

        # Plot Heading
        plt.subplot(1, 3, 3)
        plt.plot(true_heading, label='True Heading')
        plt.plot(predicted_heading, label='Predicted Heading', linestyle='dashed')
        plt.title('Heading')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def test_ai(self, input_data, segments_to_traverse, avg_points_in_segments):
        initial_sequence = []

        # Extract the initial data from the first segment path in segment_dict
        initial_data = input_data

        # Scale the initial data using the scaler
        scaled_data = initial_data.copy()
        scaled_data = self._scaler.transform(initial_data)

        # Prepare the initial sequence of data points from the scaled data
        for step in range(self._n_steps):
            initial_sequence.append(scaled_data[step])

        # Load the single model
        model = keras.models.load_model('model.keras')

        # Placeholder for all predicted data
        simulated_data_sequence = []

        # Loop through TMS data and predict for each segment
        for segment in segments_to_traverse:
            # Placeholder for predictions in the current segment
            predicted_data = []

            # Predict points for the current segment
            for _ in range(avg_points_in_segments[segment]):
                # Create the input dataset for the model
                sequence_df = pd.DataFrame(initial_sequence)
                sequence_values = sequence_df.values.astype('float32')

                prediction_input = self.create_dataset(sequence_values)
                predicted_point = model.predict(prediction_input)

                # Extract x, y, and heading from the predicted point
                x, y, heading = predicted_point[0]

                # Append predicted values with the current segment value
                predicted_data.append([x, y, heading, segment])

                # Update the sequence for the next prediction
                initial_sequence.append(predicted_point[0])
                initial_sequence.pop(0)  # Maintain the sequence length (_n_steps)

            # Convert the current segment predictions into a DataFrame
            segment_df = pd.DataFrame(
                predicted_data, columns=["x-coordinate", "y-coordinate", "heading", "segment"]
            )

            # Append the segment DataFrame to the overall simulated data sequence
            simulated_data_sequence.append(segment_df)

        # Combine all segment DataFrames into a single DataFrame
        full_simulated_data = pd.concat(simulated_data_sequence, ignore_index=True)

        # Transform the simulated data back to the original scale (if needed)
        full_simulated_data[["x-coordinate", "y-coordinate", "heading"]] = self._scaler.inverse_transform(
            full_simulated_data[["x-coordinate", "y-coordinate", "heading"]]
        )

        return full_simulated_data


    def create_dataset(self, dataset):

        sequence_data = []
        sequence_steps = []

        # Collect the last 'n' steps in reversed order for creating sequences
        for step in reversed(range(self._n_steps)):
            data_point = dataset[len(dataset) - step - 1]
            sequence_steps.append(data_point)
        
        sequence_data.append(sequence_steps)
        return np.array(sequence_data)

    def simulate_run(self, segment_paths, segment_dict):

        transformed_data_sequence = []
        initial_sequence = []

        # Extract the initial data from the first segment path in segment_dict
        initial_data = segment_dict[segment_paths[0]]

        # Scale the initial data using the scaler
        scaled_data = initial_data.copy()
        scaled_data = self._scaler.transform(initial_data)

        # Convert scaled data to a list and add it to the transformed data sequence
        scaled_data_list = scaled_data.tolist()
        for point in scaled_data_list:
            transformed_data_sequence.append(point)

        # Prepare the initial sequence of data points from the scaled data
        initial_sequence.clear()
        for step in range(self._n_steps):
            initial_sequence.append(scaled_data_list[step])

        # Calculate the number of simulated points per segment based on segment length and steps
        points_per_segment = int((len(segment_paths) * 100 - self._n_steps) / (len(segment_paths) - 1))

        # Loop over each segment, load the model, and predict new data points
        for segment_index in range(len(segment_paths) - 1):
            model_path = f'kerases/segments_{segment_paths[segment_index]}_{segment_paths[segment_index + 1]}.keras'
            model = keras.models.load_model(model_path)

            # Predict points using the loaded model
            for _ in range(points_per_segment):
                sequence_df = pd.DataFrame(initial_sequence)
                sequence_values = sequence_df.values.astype('float32')
                
                # Create dataset to be fed into the model
                prediction_input = self.create_dataset(sequence_values)
                predicted_point = model.predict(prediction_input)

                # Append the predicted point to the sequence
                initial_sequence.append(predicted_point[0])

        # Transform the simulated data back to the original scale
        simulated_data = self._scaler.inverse_transform(initial_sequence)
        return simulated_data
