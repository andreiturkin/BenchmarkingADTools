"""
Derivative Evaluation with Sympy for the following function

The Lennard-Jones potential    
def vLJ(x):
    retval = 0
    for j in range(1,N):
        for i in range(j):
            r = norm(x[i,:]-x[j,:])
            retval += r**(-12.0)-2*(r**(-6.0))
    return retval;

"""

import sympy
import numpy as np
from scipy import optimize
from LJ_Manual import Manual_vLJ_vec, Manual_dvLJ_vec
D = 3

################################
# PART 1: Computation with SYMPY
################################

def Sympy_dvLJ(x):
    N = len(x)

    xs = np.array([[sympy.Symbol('x%d%d'%(n,d)) for d in range(D)] for n in range(N)])
    # computing the function f: R^(NxD) -> R symbolically
    vLJs = 0
    for j in range(1,N):
        for i in range(j):
            rho = 0
            for d in range(D):
                rho += (xs[i,d] - xs[j,d])**2
            vLJs += rho**(-6.0)-(rho**(-3.0))
    
    # computing the gradient symbolically
    dvLJs = np.array([[sympy.diff(4*vLJs, xs[n,d]) for d in range(D)] for n in range(N)])
    
    symdict = dict()
    for n in range(N):
        for d in range(D):
            symdict[xs[n,d]] = x[n,d]
    return np.ravel(np.array([[dvLJs[n,d].subs(symdict).evalf() for d in range(D)] for n in range(N)]))
 
def Sympy_vLJ_Optimize(x):
    N = len(x)
    
    xs = np.array([[sympy.Symbol('x%d%d'%(n,d)) for d in range(D)] for n in range(N)])
    # computing the function f: R^(NxD) -> R symbolically
    vLJs = 0
    for j in range(1,N):
        for i in range(j):
            rho = 0
            for d in range(D):
                rho += (xs[i,d] - xs[j,d])**2
            vLJs += rho**(-6.0)-(rho**(-3.0))
    
    # computing the gradient symbolically
    dvLJs = np.array([[sympy.diff(4*vLJs, xs[n,d]) for d in range(D)] for n in range(N)])
    
    def Sympy_dvLJ_vec(x):
        N = len(x)/D
        retval=np.zeros(np.shape(x),dtype=float)
        x = np.reshape(x,(N,D))
        symdict = dict()
        for n in range(N):
            for d in range(D):
                symdict[xs[n,d]] = x[n,d]
        
        for n in range(N):
            for d in range(D):
                retval[n*D+d] = dvLJs[n,d].subs(symdict).evalf() 
        return retval
    
    Sympy_BFGSres = optimize.minimize(Manual_vLJ_vec, np.ravel(x),  \
                                   method='L-BFGS-B',        \
                                   jac = Sympy_dvLJ_vec,     \
                                   options={'disp': False})
    return np.reshape(Sympy_BFGSres.x, (N,D)) 