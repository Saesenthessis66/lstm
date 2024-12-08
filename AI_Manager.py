import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import joblib
import json 

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

class AI_Manager:

    _scaler_X : MinMaxScaler
    _scaler_y : MinMaxScaler
    _segment_boundaries : []

    def __init__(self, steps: int,):
        self._n_steps = steps
        self.load_segment_boundaries()

    def validate_segments(self, df):
        known_segments = set(self.segment_encoder.categories_[0])
        prediction_segments = set(df['Current segment'].unique())

        unknown_segments = prediction_segments - known_segments
        if unknown_segments:
            print(f"Warning: The following segments were not seen during training and will be ignored: {unknown_segments}")


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

        # One-hot encoding for `Current segment` with `handle_unknown='ignore'`
        self.segment_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Enable handling unknown categories
        segment_train = train_df[['Current segment']]
        segment_val = val_df[['Current segment']]

        # Fit and transform the encoder
        segment_train_encoded = self.segment_encoder.fit_transform(segment_train)
        segment_val_encoded = self.segment_encoder.transform(segment_val)

        # Scale the other features
        other_train_features = train_df[['X-coordinate', 'Y-coordinate', 'Heading']]
        other_val_features = val_df[['X-coordinate', 'Y-coordinate', 'Heading']]

        self._scaler_X = MinMaxScaler()
        X_train_scaled = self._scaler_X.fit_transform(other_train_features)
        X_val_scaled = self._scaler_X.transform(other_val_features)

        # Concatenate scaled features and one-hot-encoded segment
        X_train = np.hstack([X_train_scaled, segment_train_encoded])
        X_val = np.hstack([X_val_scaled, segment_val_encoded])

        self._scaler_y = MinMaxScaler()
        y_train = self._scaler_y.fit_transform(train_df[output_columns])
        y_val = self._scaler_y.transform(val_df[output_columns])

        # Create sequences
        def create_sequences(data, target, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(data) - seq_length):
                X_seq.append(data[i:i + seq_length])
                y_seq.append(target[i + seq_length])
            return np.array(X_seq), np.array(y_seq)

        self.X_train_seq, self.y_train_seq = create_sequences(X_train, y_train, self._n_steps)
        self.X_val_seq, self.y_val_seq = create_sequences(X_val, y_val, self._n_steps)

        # Save the OneHotEncoder and scalers
        joblib.dump(self.segment_encoder, 'segment_encoder.pkl')  # Save the encoder
        joblib.dump(self._scaler_X, 'scaler_X.pkl')
        joblib.dump(self._scaler_y, 'scaler_y.pkl')

    def train_model(self):
        # Input layer
        input_main = Input(shape=(self._n_steps, self.X_train_seq.shape[2]))

        # LSTM layers
        lstm_out = LSTM(64, return_sequences=True)(input_main)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(32, return_sequences=False)(lstm_out)

        # Dense layers
        dense_out = Dense(32, activation='relu')(lstm_out)
        dense_out = Dense(64, activation='sigmoid')(dense_out)

        # Output layer
        output_main = Dense(3)(dense_out)

        # Define the model using Model class
        model = Model(inputs=input_main, outputs=output_main)

        # Compile the model
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        # Train the model
        history = model.fit(
            self.X_train_seq, self.y_train_seq,
            validation_data=(self.X_val_seq, self.y_val_seq),
            epochs=250, batch_size=32, callbacks=[early_stopping], verbose=1
        )

        # Save the trained model
        model.save('model.keras')


    def predict_route(self, df, tms_data):

        df = pd.DataFrame(df)

        # Load scalers and model
        self._scaler_X = joblib.load('scaler_X.pkl')
        self._scaler_y = joblib.load('scaler_y.pkl')
        self.segment_encoder = joblib.load('segment_encoder.pkl')  # Load the encoder
        model = load_model('model.keras')
        
        # Use the last `n_steps` rows as initial input data
        df = df[-self._n_steps:].copy()
        predictions = []

        for idx, curr_segment in enumerate(tms_data):
            # Get segment boundaries
            boundaries = self._segment_boundaries[str(curr_segment)]
            end_coords = boundaries['end_coordinates']
            
            # Variables to track consecutive predictions moving away from endpoint
            previous_distance_to_end = None
            consecutive_moving_away = 0

            while True:
                # Prepare the latest input data
                data = df[-self._n_steps:].copy()
                segment_data = self.segment_encoder.transform(data[['Current segment']])  # One-hot encode segment
                scaled_features = self._scaler_X.transform(data[['X-coordinate', 'Y-coordinate', 'Heading']])
                
                # Concatenate scaled features with one-hot-encoded segment
                full_features = np.hstack([scaled_features, segment_data])
                input_data = np.expand_dims(full_features, axis=0)  # Add batch dimension

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

                # Check if the predicted point is close enough to the segment's endpoint
                if distance_to_end <= 0.3:
                    print(f"Reached the endpoint of Segment {curr_segment}. Moving to next segment.")
                    break

                # Check if the predicted point is moving away from the endpoint
                if previous_distance_to_end is not None:
                    if distance_to_end > previous_distance_to_end:
                        consecutive_moving_away += 1
                        print(f"Point is moving away from endpoint. Count: {consecutive_moving_away}")
                    else:
                        consecutive_moving_away = 0  # Reset if moving closer

                # Update previous distance
                previous_distance_to_end = distance_to_end

                # Switch segment if moving away for 2 consecutive points
                if consecutive_moving_away >= 2:
                    print(f"Switching to the next segment due to consecutive moving-away points.")
                    break

        # Combine all predictions into a single DataFrame
        predicted_df = pd.concat(predictions, ignore_index=True)
        return predicted_df

    def load_segment_boundaries(self):
        try:
            with open('segment_boundaries.txt', 'r') as file:
                segment_boundaries = json.load(file)
            print(f"Segment boundaries successfully read from {'segment_boundaries.txt'}")
            self._segment_boundaries = segment_boundaries
        except Exception as e:
            print(f"Error reading segment boundaries from file: {e}")