from Data_Manager import DataDivision
from AI_Manager import AI_Manager
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import keras
dataManager = DataDivision('agv.pkl')
scaler = joblib.load('scaler.pkl')

curr_segment = 56.0
df = dataManager._fullData[92:130]
# print(df)
model = keras.models.load_model('model.keras')
data = df[-10:]#n_steps


dataframe = pd.DataFrame(data)
print(data)
scaled_data = scaler.transform(data)
input_data = np.expand_dims(scaled_data, axis=0)  # Shape: (1, 10, 4)
predicted_scaled = model.predict(input_data)
placeholder = np.full((predicted_scaled.shape[0], 1), curr_segment)  # Add the current segment value
augmented_data = np.hstack((predicted_scaled, placeholder))  # Shape: (1, 4)

# Step 6: Inverse transform the predictions to get original scale
predicted_original = scaler.inverse_transform(augmented_data)

# Step 7: Remove the placeholder column (for 'Current segment') after inverse transformation
predicted_original = predicted_original[:, :3]  # Keep only the first 3 columns: X, Y, Heading

# Step 8: Convert predictions to a DataFrame for readability
result_df = pd.DataFrame(predicted_original, columns=['X-coordinate', 'Y-coordinate', 'Heading'])

# Add 'Current segment' column manually
result_df['Current segment'] = curr_segment

print(result_df)

newdata = pd.concat([data, result_df], ignore_index=True)
print(newdata)

# tms_data = [56.0,20.0,48.0,52.0,16.0,44.0,12.0,36.0,4.0]

# aiManager = AI_Manager(20)
# dataManager._fullData.drop(columns=['Timestamp'], inplace=True)
# aiManager.test_ai(dataManager._fullData[92:110],tms_data,dataManager._average_points_amount)

# aiManager.createLSTM(dataManager._day_divided_data)

# aiManager.createAI(dataManager._closePairs, dataManager._dictionaryData)

# segments_to_traverse = ['14','18','22', '26','6','10']



# toPlot = aiManager.simulate_run(segments_to_traverse, dataManager._dictionaryData)
# for s in segments_to_traverse:
#     plt.scatter(dataManager._dictionaryData[s]['X-coordinate'],dataManager._dictionaryData[s]['Y-coordinate'], color = 'green' )

# finNp = np.array(toPlot)
# plt.scatter(finNp[:, 0], finNp[:, 1], color = 'red' )

# plt.show()