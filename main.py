from Data_Manager import DataDivision
from AI_Manager import AI_Manager
import matplotlib.pyplot as plt


dataManager = DataDivision('agv.pkl')

aiManager = AI_Manager(20)
aiManager.preprocess_data(dataManager._divided_data[-1][100:-300],dataManager._divided_data[-2][100:-100])
aiManager.train_model()

tms_data = [56.0,20.0,48.0,52.0,16.0,44.0,12.0,36.0,4.0]
# tms_data = [56.0,20.0,48.0,52.0,16.0, 44.0]
df = dataManager._fullData[92:122]
avg_points_amount = dataManager._average_points_amount

predicted = aiManager.predict_route(df, tms_data, dataManager._segment_boundaries)

plt.scatter(dataManager._divided_data[0]['X-coordinate'],dataManager._divided_data[0]['Y-coordinate'])
plt.scatter(predicted['X-coordinate'],predicted['Y-coordinate'], color = 'red')
plt.show()