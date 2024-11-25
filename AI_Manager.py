import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
class AI_Manager:

    _scaler_X : MinMaxScaler
    _scaler_y : MinMaxScaler

    def __init__(self, steps: int,):
        self._n_steps = steps


    def preprocess_data(self, train_df, val_df):
        # Select features and targets
        input_columns = ['X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']
        output_columns = ['X-coordinate', 'Y-coordinate', 'Heading']

        X_train = train_df[input_columns].values
        y_train = train_df[output_columns].values
        X_val = val_df[input_columns].values
        y_val = val_df[output_columns].values

        # Initialize separate scalers
        self._scaler_X = MinMaxScaler()  # For input features
        self._scaler_y = MinMaxScaler()  # For target values

        # Fit and transform input data
        X_train = self._scaler_X.fit_transform(X_train)
        X_val = self._scaler_X.transform(X_val)

        # Fit and transform target data
        y_train = self._scaler_y.fit_transform(y_train)
        y_val = self._scaler_y.transform(y_val)

        # Create sequences
        def create_sequences(data, target, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(data) - seq_length):
                X_seq.append(data[i:i+seq_length])
                y_seq.append(target[i+seq_length])
            return np.array(X_seq), np.array(y_seq)

        self.X_train_seq, self.y_train_seq = create_sequences(X_train, y_train, self._n_steps)
        self.X_val_seq, self.y_val_seq = create_sequences(X_val, y_val, self._n_steps)
        
        joblib.dump(self._scaler_X, 'scaler_X.pkl')
        joblib.dump(self._scaler_y, 'scaler_y.pkl')


    def train_model(self):

        model = Sequential([
            LSTM(64, input_shape=(self._n_steps, self.X_train_seq.shape[2]), return_sequences=False),
            Dense(32, activation='relu'),
            Dense(64, activation='sigmoid'),
            Dense(3)  # Output layer for X, Y, Heading
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            self.X_train_seq, self.y_train_seq,
            validation_data=(self.X_val_seq, self.y_val_seq),
            epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1
        )
        model.save('model.keras')


    def predict_route(self, df, tms_data, segment_boundaries):
        """
        Predict the route using the AI model with conditions for switching to the next segment.

        Args:
            df: DataFrame containing input data.
            tms_data: List of segment identifiers.
            segment_boundaries: Dictionary of segment start and end coordinates.

        Returns:
            A DataFrame with predicted values.
        """
        # Load scalers and model
        self._scaler_X = joblib.load('scaler_X.pkl')
        self._scaler_y = joblib.load('scaler_y.pkl')
        model = load_model('model.keras')

        # Use the last `n_steps` rows as initial input data
        df = df[-self._n_steps:].copy()
        predictions = []

        print(df)

        for idx, curr_segment in enumerate(tms_data):
            # Get segment boundaries
            boundaries = segment_boundaries[curr_segment]
            start_coords = boundaries['start_coordinates']
            end_coords = boundaries['end_coordinates']

            print(f"Processing Segment {curr_segment}: Start {start_coords}, End {end_coords}")

            # Initialize variables for tracking repeated predictions and movement
            repeated_predictions = 0
            max_repeats = 5  # Threshold for identical predictions
            previous_point = None

            while True:
                # Prepare the latest input data
                data = df[-self._n_steps:].copy()
                scaled_data = self._scaler_X.transform(
                    data[['X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']].values
                )
                input_data = np.expand_dims(scaled_data, axis=0)

                # Predict the next step
                predicted_scaled = model.predict(input_data)
                predicted_original = self._scaler_y.inverse_transform(predicted_scaled)

                # Create a prediction DataFrame
                result_df = pd.DataFrame(predicted_original, columns=['X-coordinate', 'Y-coordinate', 'Heading'])
                result_df['Current segment'] = curr_segment

                # Append the prediction to the sequence and update input data
                predictions.append(result_df)
                df = pd.concat([df, result_df], ignore_index=True)

                # Analyze the predicted point
                predicted_point = result_df.iloc[-1][['X-coordinate', 'Y-coordinate']].values
                distance_to_end = np.linalg.norm(predicted_point - end_coords)
                print(f"Predicted point: {predicted_point}, Distance to endpoint: {distance_to_end:.4f}")
                if previous_point is not None:
                    previous_distance_to_end = np.linalg.norm(previous_point - end_coords)

                # Check for repeated predictions
                if previous_point is not None:
                    if np.allclose(predicted_point, previous_point, atol=0.1):
                        repeated_predictions += 1
                    elif previous_distance_to_end < distance_to_end: 
                        repeated_predictions += 1
                    else:
                        repeated_predictions = 0  # Reset if the prediction changes
                previous_point = predicted_point

                # Switch segment if repeated predictions or moving away
                if repeated_predictions >= max_repeats:
                    print(f"Switching to the next segment due to repeated predictions.")
                    break

                # Check if the predicted point is close enough to the segment's endpoint
                if distance_to_end <= 0.3:
                    print(f"Reached the endpoint of Segment {curr_segment}. Moving to next segment.")
                    break

        # Combine all predictions into a single DataFrame
        predicted_df = pd.concat(predictions, ignore_index=True)
        return predicted_df


