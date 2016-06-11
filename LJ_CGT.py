"""
Derivative Evaluation with CGT for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""
import cgt
import numpy as np
from scipy import optimize

D = 3
###################################
# PART 2c: Computation with CGT
###################################
def CGT_dvLJ(x):
    N = len(x)
    xt = cgt.vector('xt')
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((xt[i*D:i*D+D] - xt[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0))
    
    dvLJc = cgt.grad(4*vLJt, xt)    
    df = cgt.function([xt],dvLJc)
    return df(np.ravel(x))

def CGT_vLJ_Optimize(x):
    N = len(x)
    #cgt.set_precision('double')
    xt = cgt.vector('xt')
    vLJt = 0
    for j in range(1,N):
        for i in range(j):
            rho = ((xt[i*D:i*D+D] - xt[j*D:j*D+D])**2).sum()
            vLJt += rho**(-6.0)-(rho**(-3.0))
    
    f = cgt.function([xt],4*vLJt)
    dvLJc = cgt.grad(4*vLJt, xt)    
    df = cgt.function([xt],dvLJc)
    
    CGT_BFGSres = optimize.minimize(f, np.ravel(x), \
                                  method='L-BFGS-B',        \
                                  jac = df,     \
                                  options={'disp': False})
    return np.reshape(CGT_BFGSres.x, (N,D))