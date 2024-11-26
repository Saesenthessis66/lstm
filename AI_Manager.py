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
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
class AI_Manager:

    _scaler_X : MinMaxScaler
    _scaler_y : MinMaxScaler

    def __init__(self, steps: int,):
        self._n_steps = steps

    def preprocess_data(self, train_df, val_df):
        input_columns = ['X-coordinate', 'Y-coordinate', 'Heading', 'Current segment']
        output_columns = ['X-coordinate', 'Y-coordinate', 'Heading']

        # Separate `Current segment` for encoding
        segment_train = train_df[['Current segment']]
        segment_val = val_df[['Current segment']]

        # One-hot encoding for `Current segment`
        self.segment_encoder = OneHotEncoder(sparse_output=False)
        segment_train_encoded = self.segment_encoder.fit_transform(segment_train)
        segment_val_encoded = self.segment_encoder.transform(segment_val)
        joblib.dump(self.segment_encoder, 'segment_encoder.pkl')  # Save the encoder    
        # Scale other features
        other_train_features = train_df[['X-coordinate', 'Y-coordinate', 'Heading']]
        other_val_features = val_df[['X-coordinate', 'Y-coordinate', 'Heading']]

        self._scaler_X = MinMaxScaler()
        X_train_scaled = self._scaler_X.fit_transform(other_train_features)
        X_val_scaled = self._scaler_X.transform(other_val_features)

        # Concatenate scaled features and encoded segments
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


    def predict_route(self, df, tms_data, segment_boundaries):
        # Load scalers and model
        self._scaler_X = joblib.load('scaler_X.pkl')
        self._scaler_y = joblib.load('scaler_y.pkl')
        self.segment_encoder = joblib.load('segment_encoder.pkl')  # Save and load the encoder
        model = load_model('model.keras')

        # Use the last `n_steps` rows as initial input data
        df = df[-self._n_steps:].copy()
        predictions = []

        for idx, curr_segment in enumerate(tms_data):
            # Get segment boundaries
            boundaries = segment_boundaries[curr_segment]
            start_coords = boundaries['start_coordinates']
            end_coords = boundaries['end_coordinates']

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

                # Check distance to endpoint and segment transition logic (unchanged)
                predicted_point = result_df.iloc[-1][['X-coordinate', 'Y-coordinate']].values
                distance_to_end = np.linalg.norm(predicted_point - end_coords)
                if distance_to_end <= 0.3:
                    print(f"Reached the endpoint of Segment {curr_segment}. Moving to next segment.")
                    break

        # Combine all predictions into a single DataFrame
        predicted_df = pd.concat(predictions, ignore_index=True)
        return predicted_df


