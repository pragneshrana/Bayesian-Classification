import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('default')

n1 = 12
n2 = 21

x1 = np.linspace(0, 10, n1)
print('x1: ', x1)
x2 = np.linspace(0, 20, n2)
print('x2: ', x2)

X1, X2 = np.meshgrid(x1, x2)

Z = X1*X2
print('Z: ', Z.shape)



plt.figure()
plt.contour(x1, x2, Z)
                 
                 # cmap='seismic',
                 # colors='k',
               

plt.show()