import heapq
import queue
import tensorflow as tf
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
from enum import Enum 
import matplotlib.pyplot as plt

n_steps = 20

def create_dataset(dataset):
    data = []
    temp = []
    for j in reversed(range(n_steps)):
        a = dataset[len(dataset) - j - 1]
        temp.append(a)
    data.append(temp)
    return np.array(data)

import divide

_data = []
finData =[]
model: keras.Model = keras.models.load_model(r'kerases/segments_56.0_20.0.keras')

data = divide.read_data_from_csv()

segments = divide.get_unique_segments(data)

segmented_df = divide.create_interpolated_dataframes(data, segments)

close_segments = divide.find_close_segments(segmented_df, segments, distance_threshold=2.0)

segment_dict = divide.create_segment_dictionary(segments, segmented_df)

df_first = segment_dict['56.0']
df_second = segment_dict['20.0']

df = pd.concat([df_first, df_second], ignore_index=True)

_scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[['X-coordinate', 'Y-coordinate', 'Heading']] = _scaler.fit_transform(df[['X-coordinate',
                                                                                    'Y-coordinate', 'Heading']])
to_drive = df_scaled.values.tolist()
for i in range(len(to_drive)):
    _data.append(to_drive[i])


df.plot(kind = 'scatter', x = 'X-coordinate', y = 'Y-coordinate')
# plt.show()
# dataNP = np.array(_data)
# plt.scatter(dataNP[:, 0], dataNP[:, 1])
# plt.show()

# for i in range(5):
#     finData.append(to_drive[i])
# finNp = np.array(finData)
# plt.plot(finNp[:, 0], finNp[:, 1])
# plt.show()

finData.clear()

for i in range(n_steps):
    finData.append(to_drive[i])

for i in range(170):
    df2 = pd.DataFrame(finData, columns=['X-coordinate', 'Y-coordinate', 'Heading'])
    df2 = df2.values
    df2 = df2.astype('float32')
    toPredict = create_dataset(df2)
    predicted = model.predict(toPredict)
    finData.append(predicted[0])

toPlot = _scaler.inverse_transform(finData)

finNp = np.array(toPlot)
plt.plot(finNp[:, 0], finNp[:, 1], color = 'red')
plt.show()

