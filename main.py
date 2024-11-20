from Data_Manager import DataDivision
from AI_Manager import AI_Manager
import matplotlib.pyplot as plt


dataManager = DataDivision('agv.pkl')

aiManager = AI_Manager(20)
aiManager.createLSTM(dataManager._divided_data)

tms_data = [56.0,20.0,48.0,52.0,16.0,44.0,12.0,36.0,4.0]
df = dataManager._fullData[91:112]
avg_points_amount = dataManager._average_points_amount

predicted = aiManager.predict_route(df, tms_data, avg_points_amount)

plt.scatter(dataManager._divided_data[0]['X-coordinate'],dataManager._divided_data[0]['Y-coordinate'])
plt.scatter(predicted['X-coordinate'],predicted['Y-coordinate'], color = 'red')
plt.show()