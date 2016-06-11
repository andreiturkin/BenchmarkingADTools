"""
Derivative Evaluation with CasADi for the following function

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
#CasADi
from casadi import *
import numpy as np
from scipy import optimize

D = 3

###################################
# PART 2f: Computation with CasADi
###################################
def CasADi_vLJ_vec(x,arg):
    return np.array(arg[0].call([x]))
def CasADi_dvLJ_vec(x,arg):
    return np.array(arg[1].call([x.ravel()])[0]).flatten()
def CasADi_vLJ_Optimize(x):
    N = len(x)
    xc = MX.sym('xc',1,x.size)
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = 0.0
            for d in range(D):
                rho += (xc[i*D+d] - xc[j*D+d])**2
            vLJ += rho**(-6.0)-(rho**(-3.0))
    F = MXFunction('F',[xc],[4*vLJ])
    J = F.jacobian()
    
    CasADi_BFGSres = optimize.minimize(CasADi_vLJ_vec, np.ravel(x),      \
                                      method='L-BFGS-B', args=[F,J],\
                                      jac = CasADi_dvLJ_vec,         \
                                      options={'disp': False})
    return np.reshape(CasADi_BFGSres.x, (N,D))

def CasADi_dvLJ(x):
    N = len(x)
    xc = MX.sym('xc',1,x.size)
    vLJ = 0.0
    for j in range(1,N):
        for i in range(j):
            rho = 0.0
            for d in range(D):
                rho += (xc[i*D+d] - xc[j*D+d])**2
            vLJ += rho**(-6.0)-(rho**(-3.0))
    F = MXFunction('F',[xc],[4*vLJ])
    J = F.jacobian()
    return np.array(J.call([x.ravel()])[0]).flatten()
