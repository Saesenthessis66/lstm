from Data_Manager import DataDivision
from AI_Manager import AI_Manager
import numpy as np
import matplotlib.pyplot as plt

dataManager = DataDivision('michal.csv')

aiManager = AI_Manager(20)

# aiManager.createAI(dataManager._closePairs, dataManager._dictionaryData)

segments_to_traverse = ['14','18','22', '26','10']

toPlot = aiManager.simulate_run(segments_to_traverse, dataManager._dictionaryData)

finNp = np.array(toPlot)
plt.scatter(finNp[:, 0], finNp[:, 1], color = 'red' )
plt.show()