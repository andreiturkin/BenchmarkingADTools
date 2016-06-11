"""
Derivative Evaluation with Theano for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""

####################################
#Automatic Differentiation Tools
####################################
#AD
from ad import gh

import numpy as np
from scipy import optimize

D = 3
###################################
# PART 2e: Computation with AD
###################################

def AD_vLJ_vec(x):
    N = len(x)/D
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = ((x[i*D:i*D+D] - x[j*D:j*D+D])**2).sum()
            vLJ += rho**(-6.0)-(rho**(-3.0))
    return 4*vLJ

def AD_dvLJ(x):
    AD_vLJ_gradient = gh(AD_vLJ_vec)[0]
    return AD_vLJ_gradient(np.ravel(x))
 
def AD_vLJ_Optimize(x):
    N = len(x)
    AD_vLJ_gradient = gh(AD_vLJ_vec)[0]
    
    AD_BFGSres = optimize.minimize(AD_vLJ_vec, np.ravel(x),  \
                                   method='L-BFGS-B',        \
                                   jac = AD_vLJ_gradient,     \
                                   options={'disp': False})
    return np.reshape(AD_BFGSres.x, (N,D)) 