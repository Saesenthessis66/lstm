import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import divide


if __name__ == '__main__':

    data = divide.read_data_from_csv()

    segments = divide.get_unique_segments(data)

    segmented_df = divide.create_interpolated_dataframes(data, segments)

    close_segments = divide.find_close_segments(segmented_df, segments, distance_threshold=2.0)

    segment_dict = divide.create_segment_dictionary(segments, segmented_df)

    for i in range (0,len(close_segments)):

        first_segment = close_segments[i][0]
        second_segment = close_segments[i][1]

        # first_segment = 36.0
        # second_segment = 4.0

        df_first = segment_dict[str(first_segment)]
        df_second = segment_dict[str(second_segment)]

        df = pd.concat([df_first, df_second], ignore_index=True)

        # Normalize the data 
        scaler = joblib.load('scaler.pkl')
        df_scaled = df.copy()
        df_scaled = scaler.transform(df)

        # Create sequences (using df_scaled for features and df_target_scaled for targets)
        def create_sequences(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:i+n_steps])
                y.append(data[i+n_steps])
            return np.array(X), np.array(y)

        n_steps = 20
        X, y = create_sequences(df_scaled, n_steps)

        model = Sequential([
            LSTM(5, input_shape=(n_steps, 3)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
        model.compile(optimizer=optimizer, loss='mae')

        # Train the model
        model.fit(X, y, epochs=1000, batch_size=32, verbose=1)

        model.save('kerases/segments_' + str(first_segment) + '_' + str(second_segment) + '.keras')