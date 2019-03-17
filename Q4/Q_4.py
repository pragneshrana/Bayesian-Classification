import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Loading Dataset
dataset = pd.read_csv('Dataset_5_Team_41.csv')
mean_file  = open("mean_file.txt","a+")
varaince_file = open("varaince_file.txt","w+")



# #Plotting data points
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# z = dataset['Class_label']
# x1 = dataset['x_1']
# x2 = dataset['x_2']
# ax.scatter(x1, x2, z, c='r', marker='o')
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Y')
# plt.show()

mu_0 = -1  #Given 
n= [1]
sigma_sigma0_ratio =np.array([0.1,1,10,100])
sigma = np.var(dataset)
sigma_0 =[]
for i in range(len(sigma_sigma0_ratio)):
    sigma_0.append(np.divide(sigma_sigma0_ratio[i],sigma))
#Summation
dataset_length = len(dataset)
# sigma = list(np.linspace(-10e-10, 10e10, 10))
density= []
file_result = open("q_4_result.txt","a+")
###Sigma and sigma_0 are square 
for k in range(len(sigma_0)):
    for j in range(len(n)):
        dataset = dataset.sample(n=1000)     #Ramdonly selecting sample 
        x= list(dataset.iloc[:,0])
        summation = dataset.sum(axis = 0, skipna = True) 
        sigma_sq = sigma**2
        n_sigma_0 = n[j] * sigma_0[k]**2
        mu = (summation/n[j]) * (n_sigma_0/(n_sigma_0 + sigma )) + (np.square(sigma)/(n_sigma_0+np.square(sigma)))
        sigma_n = np.square(sigma*sigma_0[k])
        sigma = np.square(sigma_0[k]*sigma) / (n[j]*np.square(sigma_0[k])+sigma_sq)
        for l in range(len(x)):
            term_1 = 1/np.sqrt(2 * np.pi * np.square(sigma))
            term_2 = np.exp(-(x[l])/(2*np.square(sigma)))
            density.append(term_1 * term_2)
        plt.plot(x,density, 'bo')  # plot x and y using blue circle markers
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Density Plot')
        plt.show()
file_result.close()

    






