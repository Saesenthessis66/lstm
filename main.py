from NeuralNetwork.Data_Manager import DataManager
from NeuralNetwork.AI_Manager import AI_Manager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

# file = 'agv.pkl'
file = "wednesday.csv"
# file = 'upd_michal.csv'

dataManager = DataManager(file)

data = dataManager._fullData[:7440]
data.to_csv('trimmed_'+file)

subset1 = data[:int(len(data)*0.8)] 
subset2 = data[int(len(data)*0.8):]

aiManager = AI_Manager(25)
aiManager.train_battery_model(dataManager._fullData)
aiManager.preprocess_data(subset1, subset2)
aiManager.train_model()

# Full route, wednesday
tms_data = [258., 160., 156.,   6.,  10., 320., 316.,  24.,  20.,  54.,  58.,  56.,  52.,  22.,
  26., 318., 322.,   8.,   4., 158., 162., 256. ,252. , 62. , 66., 222., 226. ,230.,
 234., 328., 324. ,246. ,250. ,248. ,244., 326. ,330. ,286. ,290. ,334. ,338. ,238.,
 242. ,240. ,236. ,336., 332. ,118. ,122. ,262. ,266. , 30. , 34. , 64. , 60. ,254.,
 102. ,106. ,150.]

#agv pkl
# tms_data = [56.0,20.0,48.0,52.0,16.0,44.0,12.0,36.0,4.0]
# tms_data = [7.0, 39.0, 15.0, 47.0, 19.0, 55.0, 51.0, 27.0, 63.0]

# #wednesday csv
# tms_data=[56.0,52.0,22.0,26.0,318.0,322.0,8.0,4.0,158.0,162.0,256.0,252.0]
# tms_data=[240.0,236.0,336.0,332.0,118.0,122.0,262.0,266.0,30.0,34.0,64.0,60.0,254.0,258.0,102.0,106.0,150.0,154.0,206.0,210.0,208.0,204.0]

# #michal csv
# tms_data=[14.0,18.0,22.0,26.0,6.0,10.0,14.0,18.0,22.0,26.0,6.0,10.0]
# tms_data=[24.0,20.0,16.0,12.0,8.0,4.0]

#agv pkl
# df = dataManager._fullData[22:48]
# df = dataManager._fullData[140:166]

# #wednesday csv                           
# df = dataManager._fullData[949:975]
# df = dataManager._fullData[1326:1352]

#full wednesday route
df = dataManager._fullData[23:49]

# #michal csv
# df = dataManager._fullData[0:26]
# df = dataManager._fullData[207:233]


predicted = aiManager.predict_route(df, tms_data)

#agv pkl
# plt.scatter(dataManager._fullData[22:140]['X-coordinate'],dataManager._fullData[22:140]['Y-coordinate'])
# plt.scatter(dataManager._fullData[140:230]['X-coordinate'],dataManager._fullData[140:230]['Y-coordinate'])

# plt.scatter(dataManager._fullData[757:886]['X-coordinate'],dataManager._fullData[757:886]['Y-coordinate'])
# plt.scatter(dataManager._fullData[81:187]['X-coordinate'],dataManager._fullData[81:187]['Y-coordinate'])

# plt.scatter(dataManager._fullData[0:30]['X-coordinate'],dataManager._fullData[0:30]['Y-coordinate'])
# plt.scatter(dataManager._fullData[207:240]['X-coordinate'],dataManager._fullData[207:240]['Y-coordinate'])

plt.plot(dataManager._fullData[23:777]['X-coordinate'], 
         dataManager._fullData[23:777]['Y-coordinate'], 
         color='blue', 
         marker='o',  # Use a circle marker for points
         markersize=2,  # Size of the markers
         linewidth=0.5,  # Width of the connecting line
         label='Full Data')

# Highlight the predicted points in red
plt.scatter(predicted['X-coordinate'], 
            predicted['Y-coordinate'], 
            color='red', 
            label='Predicted Points')

# Add labels, legend, and grid for better visualization
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Full Data with Predicted Points')
plt.legend()
plt.grid(True)
plt.show()

results = []
with open('Config/AI/segment_boundaries.txt', 'r') as f:
    segment_boundaries = json.load(f)


    # Group data by 'Current segment'
grouped = predicted.groupby('Current segment')
first_segment = predicted['Current segment'][0]

for segment, group in grouped:
    # Get start and end coordinates for the segment
    segment_key = str(segment)  # Convert to string to match keys in the segment_boundaries
    if segment_key in segment_boundaries:
        start_coords = np.array(segment_boundaries[segment_key]["start_coordinates"])
        end_coords = np.array(segment_boundaries[segment_key]["end_coordinates"])
        
        # Get the first, second, and last points in the segment
        first_point = group.iloc[0][['X-coordinate', 'Y-coordinate']].values
        second_point = group.iloc[1][['X-coordinate', 'Y-coordinate']].values if len(group) > 1 else None
        last_point = group.iloc[-1][['X-coordinate', 'Y-coordinate']].values
        
        # Compute errors
        first_error = np.linalg.norm(first_point - start_coords)  # Euclidean distance
        second_error = np.linalg.norm(second_point - start_coords) if second_point is not None else None
        last_error = np.linalg.norm(last_point - end_coords)
        
        if not first_error == None and  not second_error == None:
            if second_error < first_error:
                first_error = second_error

        if segment == first_segment:
            first_error = '-'
        
        # Store the results
        results.append({
            'Segment': segment,
            'First point error': first_error,
            'Last point error': last_error
        })
    
# Convert results to a DataFrame
print( pd.DataFrame(results))