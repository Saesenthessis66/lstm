from Data_Manager import DataDivision
from AI_Manager import AI_Manager
import numpy as np
import matplotlib.pyplot as plt

dataManager = DataDivision('agv.pkl')
aiManager = AI_Manager(20)
segments_to_traverse = ['56.0','20.0','48.0', '52.0','16.0','44.0', '12.0', '36.0', '4.0']

toPlot = aiManager.simulate_run(segments_to_traverse, dataManager._dictionaryData)

finNp = np.array(toPlot)
plt.plot(finNp[:, 0], finNp[:, 1], color = 'red')
plt.show()