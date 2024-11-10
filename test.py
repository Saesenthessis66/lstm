import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
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
model2: keras.Model = keras.models.load_model(r'kerases/segments_20.0_48.0.keras')
# model3: keras.Model = keras.models.load_model(r'kerases/segments_48.0_52.0.keras')
# model4: keras.Model = keras.models.load_model(r'kerases/segments_52.0_16.0.keras')
# model5: keras.Model = keras.models.load_model(r'kerases/segments_16.0_44.0.keras')
# model6: keras.Model = keras.models.load_model(r'kerases/segments_44.0_12.0.keras')
# model7: keras.Model = keras.models.load_model(r'kerases/segments_12.0_36.0.keras')
# model8: keras.Model = keras.models.load_model(r'kerases/segments_36.0_4.0.keras')

data = divide.read_data_from_csv()

segments = divide.get_unique_segments(data)

segmented_df = divide.create_interpolated_dataframes(data, segments)

close_segments = divide.find_close_segments(segmented_df, segments, distance_threshold=2.0)

segment_dict = divide.create_segment_dictionary(segments, segmented_df)

df_first = segment_dict['56.0']
df_second = segment_dict['20.0']
df_third = segment_dict['48.0']

df = pd.concat([df_first, df_second, df_third], ignore_index=True)

plt.plot(df['X-coordinate'],df['Y-coordinate'], 'o')

_scaler = joblib.load('scaler.pkl')
df_scaled = df.copy()
df_scaled = _scaler.transform(df)

to_drive = df_scaled.tolist()
for i in range(len(to_drive)):
    _data.append(to_drive[i])

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

for i in range(800):
    df2 = pd.DataFrame(finData, columns=['X-coordinate', 'Y-coordinate', 'Heading'])
    df2 = df2.values
    df2 = df2.astype('float32')
    toPredict = create_dataset(df2)
    if i < 150:
        predicted = model.predict(toPredict)
        finData.append(predicted[0])
    if i >= 150 and i < 250:
        predicted = model2.predict(toPredict)
        finData.append(predicted[0])
    # if i >= 90 and i < 100:
    #     predicted = model3.predict(toPredict)
    #     finData.append(predicted[0])
    # if i >= 100 and i < 115:
    #     predicted = model4.predict(toPredict)
    #     finData.append(predicted[0])
    # if i >= 115 and i < 130:
    #     predicted = model5.predict(toPredict)
    #     finData.append(predicted[0])
    # if i >= 130 and i < 145:
    #     predicted = model6.predict(toPredict)
    #     finData.append(predicted[0])
    # if i >= 145 and i < 160:
    #     predicted = model7.predict(toPredict)
    #     finData.append(predicted[0])  
    # if i >= 160:
    #     predicted = model8.predict(toPredict)
    #     finData.append(predicted[0])      

toPlot = _scaler.inverse_transform(finData)

# finNp = np.array(toPlot)
# plt.plot(finNp[:, 0], finNp[:, 1], color = 'red')
# # plt.show()

finNp = np.array(toPlot)
plt.plot(finNp[:, 0], finNp[:, 1], color = 'red')
plt.show()