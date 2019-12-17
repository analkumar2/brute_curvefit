#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__           = "Anal Kumar"
__copyright__        = "Copyright 2019-, Anal Kumar"
__version__          = "0.0.3"
__maintainer__       = "Anal Kumar"
__email__            = "analkumar2@gmail.com"

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def bruteforce(func, x, y,  restrict, ntol = 1000, returnnfactor = 0.01):
    '''
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    ntol is the number of times the func will be run with different values of the parameters.
    returnnfactor is the fraction of the random models which will be returned in increasing order of 'fitness'

    returns the ntol*returnnfactor best models, and their errors
    '''
    numarguements = func.__code__.co_argcount - 1
    restrict = np.array( restrict)
    returnn = int(ntol*returnnfactor)
    ymin = np.min(y)
    ymax = np.max(y)
    ynorm = (y)/(ymax-ymin)
    paramlist = []
    errorlist = []
    for k in np.arange(ntol):
        currparam = []
        for i in np.arange(numarguements):
            currparam.append(np.random.rand(1)*( restrict[1,i]- restrict[0,i]) +  restrict[0,i])
        error = np.sum((func(x,*currparam)-ymin-y)**2)
        paramlist.append(currparam)
        errorlist.append(error)
        print(k/ntol,end='\r')

    best_error_idx = np.argsort(errorlist)[:returnn]
    best_params = np.array(paramlist)[best_error_idx]
    # best_error_idx = np.array(errorlist).argmin()
    # best_param = paramlist[best_error_idx]
    return [best_params, np.array(errorlist)[best_error_idx]]

def scipy_fit(func, x, y,  restrict, p0list):
    '''
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    p0list is the initial values around which the local minima will be find out by this function. Give many such values and the function will calculate local minima around all those values.

    returns the best models, and its error
    '''
    fitparams_list=[]
    error_list=[]
    ymin = np.min(y)
    ymax = np.max(y)
    ynorm = (y - ymin)/(ymax-ymin)
    def funcnorm(*args):
        return (func(*args)-ymin)/(ymax-ymin)
    for k,p0 in enumerate(p0list):
        p0 = np.ravel(p0)
        fittedparam,cov = curve_fit(func, x, y,  bounds= restrict, p0=p0)
        error = np.sum((func(x,*fittedparam)-y)**2)
        fitparams_list.append(fittedparam)
        error_list.append(error)
        print(k/len(p0list),end='\r')
    best_error_idx = np.array(error_list).argmin()
    best_param = np.array(fitparams_list)[best_error_idx]
    return [best_param, np.array(error_list)[best_error_idx]]

def brute_then_scipy(func, x, y,  restrict, ntol = 1000, returnnfactor = 0.01):
    '''
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    ntol is the number of times the func will be run with different values of the parameters.
    returnnfactor is the fraction of the random models which will be returned in increasing order of 'fitness'

    returns the best model, and its error
    '''
    paramsfitted,errors = bruteforce(func,x,y, restrict= restrict,ntol=ntol,returnnfactor=returnnfactor)
    paramfitted,error = scipy_fit(func,x,y, restrict= restrict, p0list=paramsfitted)
    return [paramfitted,error]


if __name__ == '__main__':
    def h(v, vhalf, k):
        return 1/(1+np.exp((v-vhalf)/-k))
    v = np.linspace(-0.100,0.100,3000)
    hinf = h(v,-0.050,-0.004)
    plt.plot(v,hinf, label='original')
    paramsfitted,errors = bruteforce(h,v,hinf, restrict=[[-1,-1],[1,1]])
    for param in paramsfitted:
        plt.plot(v,h(v,*param), label='fitted')
    plt.legend()
    plt.show()

    plt.plot(v,hinf, label='original')
    paramfitted,error = scipy_fit(h,v,hinf, restrict=[[-1,-1],[1,1]], p0list=paramsfitted)
    plt.plot(v,h(v,*paramfitted), label='fitted')
    plt.legend()
    plt.show()
