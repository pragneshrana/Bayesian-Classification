import numpy as np
import matplotlib.pyplot as plt

# Contour plot of bivariate normal pdf

vals= np.arange(-4,4,0.1)
X, Y =np.meshgrid(vals,vals)

Z= np.zeros(X.shape)
cov = np.array([[1.,rho],[rho,1]])
cov_inv = np.linalg.inv(cov)
det = np.linalg.det(cov)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        point = np.array([X[i,j],Y[i,j]])
        pdf = np.exp(-(np.dot(point,np.dot(cov_inv, point)))/2.)/( 2*np.pi*np.sqrt(det) )
        print('pdf: ', pdf)
        Z[i,j]=pdf
print('Z: ', Z)

        
fig =  plt.figure(figsize=(15,15))
ax = fig.gca()

plt.axis('equal')
plt.xlabel('x_1', fontsize=20)
plt.ylabel('x_2', fontsize=20)
plt.title('Contour plot of Gaussian with rho='+str(rho), fontsize=20)
plt.contourf(X,Y,Z, levels=np.arange(0.001,Z.max()+0.01,0.01) )
plt.plot([-4,4],[-4*np.sign(rho),4*np.sign(rho)],'r-', lw=2)

a=2.1

plt.plot([a,a],[-4,4], 'b-', lw =2 )
plt.plot([-4,4],[-rho*4,rho*4],'g-', lw=2)

# plt.plot([-rho*4,rho*4], [-4,4],'g-', lw=2)
# plt.plot([-4,4],[a,a], 'b-', lw =2 )

plt.colorbar()        
        
