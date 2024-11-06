import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

        df_first = segment_dict[str(first_segment)]
        df_second = segment_dict[str(second_segment)]

        df = pd.concat([df_first, df_second], ignore_index=True)

        # plt.plot(df['X-coordinate'],df['Y-coordinate'])
        # plt.title('segments_' + str(first_segment) + '_' + str(second_segment))
        # plt.show()

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

        n_steps = 20
        X, y = create_sequences(df_scaled[[ 'X-coordinate', 'Y-coordinate','Heading' ]],
                                df_target_scaled, n_steps)


        model = Sequential([
            LSTM(5, input_shape=(n_steps, 3)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, clipvalue=1.1)
        model.compile(optimizer=optimizer, loss='mse')

        # Train the model
        model.fit(X, y, epochs=500, batch_size=32, verbose=1)

        model.save('kerases/segments_' + str(first_segment) + '_' + str(second_segment) + '.keras')