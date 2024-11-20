import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import joblib
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

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
        sequence_length = 20  # Number of timesteps in each sequence
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
            LSTM(16, input_shape=(sequence_length, len(features)), return_sequences=False),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)  # Output layer for X, Y, Heading
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        model.summary()

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=250,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=2
        )

        model.save('model.keras')

        # Evaluate on validation set
        val_loss, val_mae = model.evaluate(X_val, y_val, verbose=2)
        print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")

        # # Make predictions
        # predictions = model.predict(X_val)

        # # Add a dummy column for 'Current segment' since it was included in the scaler fitting
        # dummy_column = np.zeros((y_val.shape[0], 1))  # Create a column of zeros
        # predictions_with_dummy = np.hstack([predictions, dummy_column])  # Add dummy column to predictions
        # y_val_with_dummy = np.hstack([y_val, dummy_column])  # Add dummy column to true values

        # # Inverse transform using the scaler
        # predictions_original = scaler.inverse_transform(predictions_with_dummy)[:, :3]  # Extract only the original 3 columns
        # y_val_original = scaler.inverse_transform(y_val_with_dummy)[:, :3]  # Extract only the original 3 columns

        # # Extract specific columns for individual plots
        # true_x = y_val_original[:, 0]
        # predicted_x = predictions_original[:, 0]

        # true_y = y_val_original[:, 1]
        # predicted_y = predictions_original[:, 1]

        # true_heading = y_val_original[:, 2]
        # predicted_heading = predictions_original[:, 2]

        # # Plot X-coordinate
        # plt.figure(figsize=(12, 4))
        # plt.subplot(1, 3, 1)
        # plt.plot(true_x, label='True X')
        # plt.plot(predicted_x, label='Predicted X', linestyle='dashed')
        # plt.title('X-coordinate')
        # plt.legend()

        # # Plot Y-coordinate
        # plt.subplot(1, 3, 2)
        # plt.plot(true_y, label='True Y')
        # plt.plot(predicted_y, label='Predicted Y', linestyle='dashed')
        # plt.title('Y-coordinate')
        # plt.legend()

        # # Plot Heading
        # plt.subplot(1, 3, 3)
        # plt.plot(true_heading, label='True Heading')
        # plt.plot(predicted_heading, label='Predicted Heading', linestyle='dashed')
        # plt.title('Heading')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()


    def predict_route(self, df, tms_data, avg_points_amount):

        model = keras.saving.load_model("model.keras")

        # Initialize the dataframe with the last n_steps of data
        data = df[-self._n_steps:]
        df = pd.DataFrame(data)

        # Loop through the segments in tms_data
        for segment in tms_data:
            curr_segment = segment

            # Determine the number of predictions to generate for the current segment
            if curr_segment == tms_data[0]:
                leng = avg_points_amount[curr_segment] - self._n_steps
            else:
                leng = avg_points_amount[curr_segment]

            # Generate predictions for the current segment
            for _ in range(leng):
                # Prepare the input data
                data = df[-self._n_steps:]
                dataframe = pd.DataFrame(data)
                scaled_data = self._scaler.transform(dataframe)
                input_data = np.expand_dims(scaled_data, axis=0)  # Shape: (1, n_steps, features)

                # Predict the next step
                predicted_scaled = model.predict(input_data)

                # Add the current segment value to the prediction
                placeholder = np.full((predicted_scaled.shape[0], 1), curr_segment)  # Add the current segment value
                augmented_data = np.hstack((predicted_scaled, placeholder))  # Shape: (1, features+1)

                # Inverse transform to get the original scale
                predicted_original = self._scaler.inverse_transform(augmented_data)

                # Keep only the first 3 columns (X, Y, Heading) and add segment
                predicted_original = predicted_original[:, :3]  # Keep X, Y, Heading
                result_df = pd.DataFrame(predicted_original, columns=['X-coordinate', 'Y-coordinate', 'Heading'])
                result_df['Current segment'] = curr_segment  # Add segment information

                # Append the prediction to the main DataFrame
                df = pd.concat([df, result_df], ignore_index=True)

        return df