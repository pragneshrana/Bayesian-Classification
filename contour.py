import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd 



# n=5    #no of partition
# x1 = list(np.linspace(0, 100, n))
# x2 = list(np.linspace(0, 100, n))
# X, Y = np.meshgrid(x1, x2)
# print('Y: ', Y)
# print('X: ', X)
# Z = np.sin(X)+Y
# print('Z: ', Z)

# plt.contour(X, Y, Z)
# plt.show()

n=5    #no of partition
x1 = list(np.linspace(0, 100, n))
x2 = list(np.linspace(0, 100, n))
dataframe = pd.DataFrame([])

#Meshgrid
for i in range(len(x1)):
    for j in range(len(x2)):
        data_vector = [x2[j], x1[i]]
        data_vector = pd.Series(data_vector)
        dataframe = dataframe.append(data_vector, ignore_index=True)
print('dataframe: ', dataframe)
z=[]
for i in range(len(dataframe)):
  z.append(np.sin(dataframe[0][i])+dataframe[1][i])
z=np.array(z)
z=z.reshape(-1,n)
print('z: ', z)
density_array = []