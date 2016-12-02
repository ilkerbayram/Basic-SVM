# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 12:10:56 2016

@author: ilker bayram
"""

import numpy as np
import matplotlib.pyplot as plt
import SVM
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# the classification function based on the Gaussian Kernel used in this demo
def Gaussf(z, alp, x, y, sig, b):
    f = b + sum( alp * y * np.exp( -1.0 * sum( (x - z)**2 ) / ( sig**2 ) ) ) 
    return f

# below are some functions to create classes with different shapes in R^2
# the circular support function
def circleF( x, R ):
    # assigns 1 if x lies inside a circle of radius R (around the origin)
    y = np.ones(x.shape[1])
    y[(x[0,:]**2 + x[1,:]**2) > R**2 ]  = -1
    return y
    
# the diamond support function
def diamondF( x, R ):
    # assigns 1 if x lies inside a diamond of radius R (around the origin)
    y = np.ones(x.shape[1])
    y[(abs(x[0,:]) + abs(x[1,:])) > R ]  = -1
    return y

# the sinusoidal support function
def sinF( x, t, R ):
    # assigns 1 if x lies inside a diamond of radius R (around the origin)
    y = np.ones(x.shape[1])
    y[ np.sin( np.pi * x[0,:]) + t < x[1,:]  ]  = -1
    y[ np.sin( np.pi * x[0,:]) - t > x[1,:]  ]  = -1
    y[(x[0,:]**2 + x[1,:]**2) > R**2 ]  = -1
    return y
    
  
N = 2000; # number of points
x = np.random.uniform(-2,2,(2,N))
# now choose the classifier.
# All three are valid choices...

#y = circleF(x,0.5)
#y = diamondF(x,1)
y = sinF(x,1,2)

plt.plot(x[0, y == 1], x[1,y == 1], 'ro', markeredgecolor='r', markersize = 2, label = 'Class-1')
plt.plot(x[0, y == -1], x[1,y == -1], 'bo', markeredgecolor='b', markersize = 2, label = 'Class-2')
plt.legend()
plt.axis('equal')
plt.title('Available Data')
# construct the H matrix that SVMtrain_FB requires
# set the parameters
sig = 0.5 
C = 10.0
# compute H
dist = ( ( np.tile(x[0,:],(N,1)) -  np.tile(x[0,:],(N,1)).transpose() )**2 \
        + ( np.tile(x[1,:],(N,1)) -  np.tile(x[1,:],(N,1)).transpose() )**2 )
Y = np.tile(y,(N,1)) * np.tile(y,(N,1)).transpose()
H = ( np.exp(- dist / (sig**2) ) + np.identity(N) / C ) * Y 

# train!
MAX_ITER = 2000
alp = SVM.SVMdual_FB(H, y, MAX_ITER)

# only need the support vectors for classification
ind = alp > 1e-5
alpsup = alp[ind]
ysup = y[ind]
xsup = x[:,ind]

# determine b
k = np.argmax(alpsup)
b = 0
f = Gaussf(xsup[:,k].reshape((2,1)), alpsup, xsup, ysup, sig, b)
b = (1 - alpsup[k] / C ) / y[k] - f

# display the results
grid = np.arange(-3, 3, 0.05)
xx, yy = np.meshgrid(grid, grid)
t = np.concatenate((xx.reshape((1,xx.size)),yy.reshape((1,yy.size))),axis = 0)
f = np.zeros(xx.size)

for k in range(0, xx.size):
    f[k] = Gaussf(t[:,k].reshape((2,1)), alpsup, xsup, ysup, sig, b)

f = f.reshape((xx.shape[0],xx.shape[1]))


fig = plt.figure()
im = plt.imshow(f, interpolation='bilinear', origin='lower', extent=[min(grid), max(grid), min(grid), max(grid)],
                vmax=f.max(), vmin=f.min())
plt.contour(xx,yy,f,[0.0])
fig.colorbar(im, ticks = np.linspace(f.min(), f.max(), num=5))
plt.title('Decision Function f and its Zero Contour')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Decision Function f')
plt.show()