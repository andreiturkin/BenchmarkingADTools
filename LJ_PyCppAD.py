"""
Derivative Evaluation with PYADOLC for the following function

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
#PyCppAD
from pycppad import *
import numpy as np
from scipy import optimize

D = 3

###################################
# PART 2a: Optimization with PyCppAD
###################################
def PyCppAD_vLJ_for_Optimize(x,arg):
    return arg.forward(0, x) 

def PyCppAD_dvLJ_for_Optimize(x,arg):
    return arg.jacobian(x).flatten()

def PyCppAD_vLJ_Optimize(x):
    N = len(x)
    ix = np.zeros(np.ravel(x).shape)
    ad_x = independent(ix)
    # computing the function f: R^(NxD) -> R with PyCppAD
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((ad_x[i*D:i*D+D] - ad_x[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0)) 
    vLJt = np.array([4*vLJt])
    
    f =  adfun(ad_x, vLJt) 
    
    PyCppAD_BFGSres = optimize.minimize(PyCppAD_vLJ_for_Optimize, np.ravel(x),\
                                        jac=PyCppAD_dvLJ_for_Optimize, args=(f,),\
                                        method='L-BFGS-B', options={'disp': False})
    return np.reshape(PyCppAD_BFGSres.x, (N,D))

###################################
# PART 2a: Computation with PyCppAD
###################################
def PyCppAD_dvLJ(x):
    N = len(x)
    ix = np.zeros(np.ravel(x).shape)
    ad_x = independent(ix)
    # computing the function f: R^(NxD) -> R with PyCppAD
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((ad_x[i*D:i*D+D] - ad_x[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0)) 
    vLJt = np.array([4*vLJt])
    
    f =  adfun(ad_x, vLJt)
    return f.jacobian(np.ravel(x))