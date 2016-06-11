"""
Derivative Evaluation for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""

import numpy as np
from scipy import optimize
D = 3

####################################
# PART 0: Computing gradient by hand
####################################
def Manual_vLJ(x):
    N = len(x)
    retval = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = ((x[i,:] - x[j,:])**2).sum()
            retval += rho**(-6.0)-(rho**(-3.0))
    return 4*retval

def Manual_vLJ_vec(x):
    N = len(x)/D
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = ((x[i*D:i*D+D] - x[j*D:j*D+D])**2).sum()
            vLJ += rho**(-6.0)-(rho**(-3.0))
    return 4*vLJ

def Manual_dvLJ(x):
    N = len(x)
    g = np.zeros(np.shape(x),dtype=float)
    for n in range(N):
        for d in range(D):
            for m in range(N):
                if n != m:
                    g[n,d] -= 12*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**7 - 6*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**4
    return np.ravel(4*g)

def Manual_dvLJ_vec(x):
    N = len(x)/D
    x = np.reshape(x,(N,D))
    g = np.zeros(np.shape(x),dtype=float)
    for n in range(N):
        for d in range(D):
            for m in range(N):
                if n != m:
                    g[n,d] -= 12*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**7 - 6*(x[n,d] - x[m,d])/(((x[n,:]-x[m,:])**2).sum())**4
    return np.ravel(4*g)

def Manual_vLJ_Optimize(x):
    N = len(x)
    
    Manual_BFGSres = optimize.minimize(Manual_vLJ_vec, np.ravel(x),  \
                                   method='L-BFGS-B',        \
                                   jac = Manual_dvLJ_vec,     \
                                   options={'disp': False})
    return np.reshape(Manual_BFGSres.x, (N,D)) 