import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import joblib
import keras
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


def simulate_run(segments):

    _data = []
    finData =[]

    d = divide.read_data_from_csv()

    s = divide.get_unique_segments(d)

    segmented_df = divide.create_interpolated_dataframes(d, s)

    segment_dict = divide.create_segment_dictionary(s, segmented_df)

    data = segment_dict[segments[0]]

    _scaler = joblib.load('scaler.pkl')
    df_scaled = data.copy()
    df_scaled = _scaler.transform(data)

    to_drive = df_scaled.tolist()
    for i in range(len(to_drive)):
        _data.append(to_drive[i])

    finData.clear()

    for i in range(n_steps):
        finData.append(to_drive[i])

    simulated_points_amount = int((len(segments) * 100 - n_steps) / (len(segments) - 1))

    for i in range(0, len(segments)-1):
        model_name = 'kerases/segments_'+segments[0]+'_'+segments[0 + 1]+'.keras'

        model = keras.models.load_model(model_name)

        for i in range(simulated_points_amount):
            df2 = pd.DataFrame(finData)
            df2 = df2.values
            df2 = df2.astype('float32')
            toPredict = create_dataset(df2)

            predicted = model.predict(toPredict)
            finData.append(predicted[0])
    

    toPlot = _scaler.inverse_transform(finData)
    return toPlot

import divide

if __name__ == '__main__':

    toPlot = simulate_run(['56.0','20.0','48.0'])

    finNp = np.array(toPlot)
    plt.plot(finNp[:, 0], finNp[:, 1], color = 'red')
    plt.show()