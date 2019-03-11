import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Boundary Plotting
x1 = list(np.linspace(-100, 50, 50))
x2 = list(np.linspace(-100, 80, 50))
dataframe = pd.DataFrame([])

for i in range(len(x1)):
    print('i: ', i)
    for j in range(len(x2)):
        data_vector = [x1[i], x2[j]]
        data_vector = pd.Series(data_vector)
        dataframe = dataframe.append(data_vector, ignore_index=True)


# Adding decision boundary to plot

def decision_boundary(x_1):
    x_1 = np.array(x_1)
    """ Calculates the x_2 value for plotting the decision boundary."""
    return 4 - np.sqrt(-np.square(x_1) + 4*x_1 )

bound = decision_boundary(x1)
plt.plot(x1, bound, 'r--', lw=3)

# x_vec = np.linspace(*ax.get_xlim())
x_1 = np.arange(0, 100, 0.05)

plt.show()
