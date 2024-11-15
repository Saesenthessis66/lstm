import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class AI_Manager:

    _n_steps : int = 20
    _scaler = any

    def __init__(self, steps : int):
        self._n_steps = steps
        self._scaler = joblib.load('scaler.pkl')


    def createAI(self, close_segments, segment_dict):

        for i in range (0,len(close_segments)):

            # Assign two close segments IDs 
            # first_segment = close_segments[i][0]
            # second_segment = close_segments[i][1]

            first_segment = 10
            second_segment = 14

            # Load data for two close segments
            df_first = segment_dict[str(first_segment)]
            df_second = segment_dict[str(second_segment)]

            # Concatinate data for two close segments
            df = pd.concat([df_first, df_second], ignore_index=True)

            # Normalize the data 
            df_scaled = df.copy()
            df_scaled = self._scaler.transform(df)

            # Create sequences (using df_scaled for features and df_target_scaled for targets)
            def create_sequences(data, n_steps):
                X, y = [], []
                for i in range(len(data) - n_steps):
                    X.append(data[i:i+n_steps])
                    y.append(data[i+n_steps])
                return np.array(X), np.array(y)

            # Initialize steps amount for LSTM network
            self._n_steps = 20
            X, y = create_sequences(df_scaled, self._n_steps)

            model = Sequential([
                LSTM(5, input_shape=(self._n_steps, 3)),
                Dense(32, activation='relu'),
                Dense(32, activation='relu'),
                Dense(3)
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
            model.compile(optimizer=optimizer, loss='mae')

            # Train the model
            model.fit(X, y, epochs=1000, batch_size=32, verbose=1)

            model.save('kerases/segments_' + str(first_segment) + '_' + str(second_segment) + '.keras')

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
