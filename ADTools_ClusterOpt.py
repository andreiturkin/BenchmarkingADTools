"""
This Python file can be used to replicate the results obtained for:
A.Turkin "Benchmarking Python Tools for Automatic Differentiation"
Author: Andrei Turkin.
E-mail: andrei_turkin@hotmail.com
"""
####################################
#Manual and Symbolic Differentiation
####################################
from LJ_Manual import Manual_vLJ, Manual_dvLJ, Manual_vLJ_Optimize
from LJ_Sympy import Sympy_dvLJ, Sympy_vLJ_Optimize

####################################
#Automatic Differentiation Tools
####################################
from LJ_PyAdolc import PyAdolc_dvLJ, PyAdolc_vLJ_Optimize
from LJ_PyCppAD import PyCppAD_dvLJ, PyCppAD_vLJ_Optimize
from LJ_CasADi import CasADi_dvLJ, CasADi_vLJ_Optimize
from LJ_CGT import CGT_dvLJ, CGT_vLJ_Optimize
from LJ_Theano import Theano_dvLJ, Theano_vLJ_Optimize
from LJ_AD import AD_dvLJ, AD_vLJ_Optimize

####################################
#Additional Tools
####################################
from ExcelTools import SavePrecisionToExcel,SaveSpeedtoExcelFile
from DatasetTools import GetXFromDataSet

####################################
import numpy as np
from numpy.random import random
####################################
import timeit
###################################
from itertools import chain

#Number of dimensions
D = 3

def GetListOfGrads(x, iListOfGradFunc, iListOfOptFunc, possibles):
    ListOfGrads=[]
    ListOfEins=[]
    ListOfnGradins=[]
    ListOfEopts=[]
    ListOfnGrandopts=[]
    
    for i in range(len(iListOfGradFunc)):  
        optfunc = possibles.get(iListOfOptFunc[i])
        gradfunc = possibles.get(iListOfGradFunc[i])
        print 'Getting values for {} function'.format(iListOfGradFunc[i].split('_')[0])
        Eofx = Manual_vLJ(x)
        optx = optfunc(x)
        Eofoptx = Manual_vLJ(optx)
        gradnormx = np.linalg.norm(gradfunc(x))  
        gradnormoptx = np.linalg.norm(gradfunc(optx))          

        print 'Gradient norm at the initial point: {}'.format(gradnormx)
        print 'Function Value at the point: {}'.format(Eofx)
        print 'Gradient norm at the optimal point: {}'.format(gradnormoptx)
        print 'Function Value at the point: {}'.format(Eofoptx)
        
        ListOfGrads.append(gradfunc(optx))
        ListOfEins.append(Eofx)
        ListOfnGradins.append(gradnormx)
        ListOfEopts.append(Eofoptx)
        ListOfnGrandopts.append(gradnormoptx)
    return (ListOfGrads,ListOfEins,ListOfnGradins,ListOfEopts,ListOfnGrandopts) 

def PrecisionTest():
#     Uncomment if you are going to run this test for the whole bunch of tools 
#     ListOfGradFunc = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ',\
#                        'CGT_dvLJ',    'Theano_dvLJ' ,'AD_dvLJ']
#     ListOfOptFunc  = ['Manual_vLJ_Optimize', 'PyAdolc_vLJ_Optimize',\
#                        'PyCppAD_vLJ_Optimize','CasADi_vLJ_Optimize',  'CGT_vLJ_Optimize',\
#                        'Theano_vLJ_Optimize' ,'AD_vLJ_Optimize']

#    Comment the following two lines if you are going to run this test for the tools above
    ListOfGradFunc = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ']
    ListOfOptFunc  = ['Manual_vLJ_Optimize', 'PyAdolc_vLJ_Optimize', 'PyCppAD_vLJ_Optimize']
    #Find all of the functions
    possibles = globals().copy()
    possibles.update(locals())
    
    # Initialize the initial matrix of atom positions
    ListOfXs = GetXFromDataSet('Dataset/')
        
    for j in range(len(ListOfXs)):
        x = ListOfXs[j]
        N = len(x)
        print 'Initial condition: N = {}\n X = {}'.format(N,x) 
        (ListOfGrads,ListOfEins,ListOfnGradins,ListOfEopts,ListOfnGrandopts) =\
        GetListOfGrads(x, ListOfGradFunc, ListOfOptFunc, possibles)
        SavePrecisionToExcel('PrecisionTest.xlsx', ListOfGradFunc, ListOfGrads, N,\
                             ListOfEins, ListOfnGradins, ListOfEopts, ListOfnGrandopts)
    print '\nDone'
    
def SpeedTest():
    global N
    
    ListOfGradFunc1 = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ',\
                      'CGT_dvLJ',    'AD_dvLJ', 'Theano_dvLJ']
    ListOfGradFunc2 = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ', 'CGT_dvLJ', 'AD_dvLJ']
    ListOfGradFunc3 = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ', 'CGT_dvLJ']
    ListOfGradFunc4 = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ', 'CasADi_dvLJ']
    ListOfGradFunc5 = ['Manual_dvLJ','PyAdolc_dvLJ','PyCppAD_dvLJ']
    
    #Find all of the functions
    possibles = globals().copy()
    possibles.update(locals())

    # Initialize the initial matrix of atom positions
    ListOfXs = GetXFromDataSet('Dataset/')
    
    myrange = chain(range(4,20,4),range(20,400,10),range(400,len(ListOfXs),50))
    
    for N in myrange:
        x = ListOfXs[N-3]
        print '\nInitial condition: N={}\n'.format(N)
        print x
        
        if N<20:
            FunctionList = ListOfGradFunc1
            ListOfTimes1 = []
            for i in range(len(FunctionList)):
                Function = possibles.get(FunctionList[i])
                t = timeit.Timer(lambda: Function(x))
                exectime = t.timeit(number=3)/3
                print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
                ListOfTimes1.append(exectime)
            SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, ListOfTimes1)
        elif N>=20 and N<50:
            ListOfTimes2 = []
            FunctionList = ListOfGradFunc2
            for i in range(len(FunctionList)):
                Function = possibles.get(FunctionList[i])
                t = timeit.Timer(lambda: Function(x))
                exectime = t.timeit(number=3)/3
                print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
                ListOfTimes2.append(exectime)
            SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, ListOfTimes2)
        elif N>=50 and N<190:
            ListOfTimes3 = []
            FunctionList = ListOfGradFunc3
            for i in range(len(FunctionList)):
                Function = possibles.get(FunctionList[i])
                t = timeit.Timer(lambda: Function(x))
                exectime = t.timeit(number=3)/3
                print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
                ListOfTimes3.append(exectime)
            SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, ListOfTimes3)
        elif N>=190 and N<600:
            ListOfTimes4 = []
            FunctionList = ListOfGradFunc4
            for i in range(len(FunctionList)):
                Function = possibles.get(FunctionList[i])
                t = timeit.Timer(lambda: Function(x))
                exectime = t.timeit(number=3)/3
                print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
                ListOfTimes4.append(exectime)
            SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, ListOfTimes4)
        else:
            ListOfTimes5 = []
            FunctionList = ListOfGradFunc5
            for i in range(len(FunctionList)):
                Function = possibles.get(FunctionList[i])
                t = timeit.Timer(lambda: Function(x))
                exectime = t.timeit(number=3)/3
                print '{} execution time: {}'.format(FunctionList[i].split('_')[0], exectime)
                ListOfTimes5.append(exectime)
            SaveSpeedtoExcelFile('SpeedTest.xlsx', N, FunctionList, ListOfTimes5)
        
if __name__ == '__main__':
    SpeedTest()
    PrecisionTest()
    
    
