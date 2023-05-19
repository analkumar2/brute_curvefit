#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Anal Kumar"
__copyright__ = "Copyright 2019-, Anal Kumar"
__version__ = "0.0.7"
__maintainer__ = "Anal Kumar"
__email__ = "analkumar2@gmail.com"

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from time import time
from multiprocessing import Pool
import os
import sys
import pickle
import pprint


def funcnorm_par(func_args_ymin_yrange):
    func = func_args_ymin_yrange[0]
    args = func_args_ymin_yrange[1]
    ymin = func_args_ymin_yrange[2]
    yrange = func_args_ymin_yrange[3]
    tore = (func(*args) - ymin) / yrange
    # np.random.seed()
    # np.save(f'tempplots/{np.random.randint(1000)}', tore)
    return tore


def bruteforce(
    func,
    x,
    y,
    restrict,
    ntol=1000,
    returnnfactor=0.01,
    printerrors=True,
    parallel=False,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    ntol is the number of times the func will be run with different values of the parameters.
    returnnfactor is the fraction of the random models which will be returned in increasing order of 'fitness'
    savetofile to save the output to a textfile

    returns the ntol*returnnfactor best models, and their errors
    """
    if parallel:
        numarguements = func.__code__.co_argcount - 1
        restrict = np.array(restrict)
        returnn = int(ntol * returnnfactor)
        ymin = np.min(y)
        ymax = np.max(y)
        if ymax - ymin == 0:
            yrange = 1
        else:
            yrange = ymax - ymin
        ynorm = (y - ymin) / yrange
        # def funcnorm(args):
        #     return (func(*args)-ymin)/yrange
        paramlist = []
        errorlist = []
        args_list = []
        for k in np.arange(ntol):
            currparam = []
            for i in np.arange(numarguements):
                currparam.append(
                    (
                        np.random.rand(1) * (restrict[1, i] - restrict[0, i])
                        + restrict[0, i]
                    )[0]
                )
            paramlist.append(currparam)
            args_list.append([list(x), *list(currparam)])

        pool = Pool(processes=int(os.cpu_count()*2/4))
        A = pool.map(
            funcnorm_par,
            zip(
                np.tile(func, ntol),
                args_list,
                np.tile(ymin, ntol),
                np.tile(yrange, ntol),
            ),
        )
        for a in A:
            # np.save(f'tempplots/{np.random.randint(1000)}', [a,ynorm])
            error = np.sum((a - ynorm) ** 2)
            if printerrors == True:
                print(f"error = {error}")
            errorlist.append(error)
        pool.terminate()
        best_error_idx = np.argsort(errorlist)[:returnn]
        best_params = np.array(paramlist)[best_error_idx]
        # best_error_idx = np.array(errorlist).argmin()
        # best_param = paramlist[best_error_idx]
        if savetofile != False:
            with open(savetofile, "wb") as f:
                pickle.dump([best_params, np.array(errorlist)[best_error_idx]], f)

        return [best_params, np.array(errorlist)[best_error_idx]]

    else:
        numarguements = func.__code__.co_argcount - 1
        restrict = np.array(restrict)
        returnn = int(ntol * returnnfactor)
        ymin = np.min(y)
        ymax = np.max(y)
        if ymax - ymin == 0:
            yrange = 1
        else:
            yrange = ymax - ymin
        ynorm = (y - ymin) / yrange

        def funcnorm(*args):
            return (func(*args) - ymin) / yrange

        paramlist = []
        errorlist = []
        for k in np.arange(ntol):
            currparam = []
            for i in np.arange(numarguements):
                currparam.append(
                    (
                        np.random.rand(1) * (restrict[1, i] - restrict[0, i])
                        + restrict[0, i]
                    )[0]
                )
            error = np.sum((funcnorm(x, *currparam) - ynorm) ** 2)
            if printerrors == True:
                print(f"error = {error}")
            paramlist.append(currparam)
            errorlist.append(error)
        #     print(k / ntol, end="\r")
        # print("    ")

        best_error_idx = np.argsort(errorlist)[:returnn]
        best_params = np.array(paramlist)[best_error_idx]
        # best_error_idx = np.array(errorlist).argmin()
        # best_param = paramlist[best_error_idx]
        if savetofile != False:
            with open(savetofile, "wb") as f:
                pickle.dump([best_params, np.array(errorlist)[best_error_idx]], f)
        return [best_params, np.array(errorlist)[best_error_idx]]


def scipy_fit_parhelper(zippedh):
    func = zippedh[0]
    ymin = zippedh[1]
    yrange = zippedh[2]
    x = zippedh[3]
    ynorm = zippedh[4]
    restrict = zippedh[5]
    p0 = np.ravel(zippedh[6])
    maxfev = zippedh[7]
    printerrors = zippedh[8]

    def funcnorm(*args):
        return (func(*args) - ymin) / yrange

    try:
        fittedparam, cov = curve_fit(
            funcnorm, x, ynorm, bounds=restrict, p0=p0, maxfev=maxfev
        )
        error = np.sum((funcnorm(x, *fittedparam) - ynorm) ** 2)
        if printerrors == True:
            print(f"error = {error}")
            print(f"fittedparam = {fittedparam}")
    except RuntimeError:
        print("RuntimeError")
        return [[1, 2, 3], np.inf]
    return [fittedparam, error]


def scipy_fit(
    func,
    x,
    y,
    restrict,
    p0list,
    maxfev=1000,
    printerrors=True,
    parallel=False,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    p0list is the initial values around which the local minima will be find out by this function. Give many such values and the function will calculate local minima around all those values.
    maxfev is the the maximum number of calls to the function by curve_fit

    returns the best model, and its error
    """
    starttime = time()
    if parallel:
        fitparams_list = []
        error_list = []
        ymin = np.min(y)
        ymax = np.max(y)
        if ymax - ymin == 0:
            yrange = 1
        else:
            yrange = ymax - ymin
        ynorm = (y - ymin) / yrange

        def funcnorm(*args):
            return (func(*args) - ymin) / yrange

        pool = Pool(processes=int(os.cpu_count()*2/4))
        A = pool.map(
            scipy_fit_parhelper,
            zip(
                [func] * len(p0list),
                [ymin] * len(p0list),
                [yrange] * len(p0list),
                [x] * len(p0list),
                [ynorm] * len(p0list),
                [restrict] * len(p0list),
                p0list,
                [maxfev] * len(p0list),
                [printerrors] * len(p0list),
            ),
        )
        for a in A:
            fittedparam = a[0]
            error = a[1]
            fitparams_list.append(fittedparam)
            error_list.append(error)
        pool.terminate()

        best_error_idx = np.array(error_list).argmin()
        best_param = np.array(fitparams_list)[best_error_idx]
        # print('timetaken', time()-starttime)
        if savetofile != False:
            with open(savetofile, "wb") as f:
                pickle.dump([fitparams_list, error_list], f)
        return [best_param, np.array(error_list)[best_error_idx]]
    else:
        fitparams_list = []
        error_list = []
        ymin = np.min(y)
        ymax = np.max(y)
        if ymax - ymin == 0:
            yrange = 1
        else:
            yrange = ymax - ymin
        ynorm = (y - ymin) / yrange

        def funcnorm(*args):
            return (func(*args) - ymin) / yrange

        for k, p0 in enumerate(p0list):
            p0 = np.ravel(p0)
            try:
                fittedparam, cov = curve_fit(
                    funcnorm, x, ynorm, bounds=restrict, p0=p0, maxfev=maxfev
                )
                error = np.sum((funcnorm(x, *fittedparam) - ynorm) ** 2)
                if printerrors == True:
                    print(f"error = {error}")
                fitparams_list.append(fittedparam)
                error_list.append(error)
            except RuntimeError:
                print("RuntimeError")
        #     print(k / len(p0list), end="\r")
        # print("     ")
        best_error_idx = np.array(error_list).argmin()
        best_param = np.array(fitparams_list)[best_error_idx]
        # print('timetaken', time()-starttime)
        if savetofile != False:
            with open(savetofile, "wb") as f:
                pickle.dump([fitparams_list, error_list], f)
        return [best_param, np.array(error_list)[best_error_idx]]


def scipy_minimize(
    func,
    x,
    y,
    restrict,
    p0list,
    method=None,
    jac=None,
    printerrors=True,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    p0list is the initial values around which the local minima will be find out by this function. Give many such values and the function will calculate local minima around all those values.

    returns the best model, and its error
    """
    wr_restrict = Bounds(restrict[0], restrict[1])
    fitparams_list = []
    error_list = []
    ymin = np.min(y)
    ymax = np.max(y)
    if ymax - ymin == 0:
        yrange = 1
    else:
        yrange = ymax - ymin
    ynorm = (y - ymin) / yrange

    def funcnorm(*args):
        return (func(*args) - ymin) / yrange

    def wr_funcnorm(pll):
        return np.sum((funcnorm(x, *pll) - ynorm) ** 2)

    for k, p0 in enumerate(p0list):
        p0 = np.ravel(p0)
        if method == "L-BFGS-B" or method == "TNC":
            fittedy = minimize(
                wr_funcnorm, p0, method=method, jac=jac, bounds=wr_restrict
            )
        else:
            fittedy = minimize(wr_funcnorm, p0, method=method, jac=jac)
        error = fittedy.fun
        fitparams_list.append(fittedy.x)
        error_list.append(error)
    #     print(k / len(p0list), end="\r")
    # print("    ")
    best_error_idx = np.array(error_list).argmin()
    best_param = np.array(fitparams_list)[best_error_idx]
    if savetofile != False:
        with open(savetofile, "wb") as f:
            pickle.dump([fitparams_list, error_list], f)
    return [best_param, np.array(error_list)[best_error_idx]]


def brute_scifit(
    func,
    x,
    y,
    restrict,
    ntol=1000,
    returnnfactor=0.01,
    maxfev=1000,
    printerrors=True,
    parallel=False,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    ntol is the number of times the func will be run with different values of the parameters.
    returnnfactor is the fraction of the random models which will be returned in increasing order of 'fitness'

    returns the best model, and its error
    """
    savetofilebf = 'bf_'+savetofile if (savetofile != False) else savetofile
    savetofilesf = 'sf_'+savetofile if (savetofile != False) else savetofile
    paramsfitted, errors = bruteforce(
        func,
        x,
        y,
        restrict=restrict,
        ntol=ntol,
        returnnfactor=returnnfactor,
        printerrors=printerrors,
        parallel=parallel,
        savetofile=savetofilebf,
    )
    paramfitted, error = scipy_fit(
        func,
        x,
        y,
        restrict=restrict,
        p0list=paramsfitted,
        maxfev=maxfev,
        printerrors=printerrors,
        parallel=parallel,
        savetofile=savetofilesf,
    )
    return [paramfitted, error]


def brute_then_scipy(
    func,
    x,
    y,
    restrict,
    ntol=1000,
    returnnfactor=0.01,
    maxfev=1000,
    printerrors=True,
    parallel=False,
    savetofile=False,
):
    """
    Alternate name for brute_scifit. Included for backward compatibility.
    """
    return brute_scifit(
        func,
        x,
        y,
        restrict,
        ntol=1000,
        returnnfactor=0.01,
        maxfev=1000,
        printerrors=printerrors,
        parallel=parallel,
    )


def brute_scimin(
    func,
    x,
    y,
    restrict,
    method="TNC",
    ntol=1000,
    returnnfactor=0.01,
    jac=None,
    printerrors=True,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    ntol is the number of times the func will be run with different values of the parameters.
    returnnfactor is the fraction of the random models which will be returned in increasing order of 'fitness'

    returns the best model, and its error
    """
    paramsfitted, errors = bruteforce(
        func,
        x,
        y,
        restrict=restrict,
        ntol=ntol,
        returnnfactor=returnnfactor,
        printerrors=printerrors,
        savetofile=savetofile,
    )
    paramfitted, error = scipy_minimize(
        func,
        x,
        y,
        restrict,
        p0list=paramsfitted,
        jac=jac,
        method=method,
        printerrors=printerrors,
        savetofile=savetofile,
    )
    return [paramfitted, error]


def scipy_bashop(
    func,
    x,
    y,
    restrict,
    p0list,
    method=None,
    jac=None,
    printerrors=True,
    savetofile=False,
):
    """
    func is any function with the first arguemnet the list of Independent variable and all the next variables free parameters.
    x is the list of values of the Independent variable. y is the actual data to be fitted to.
    restrict is list of two lists of the form [[minA, minB, minC],[maxA, maxB, maxC]] where minI and maxI are the minimum and maximum value parameter I can take
    p0list is the initial values around which the local minima will be find out by this function. Give many such values and the function will calculate local minima around all those values.

    returns the best model, and its error
    """
    wr_restrict = Bounds(restrict[0], restrict[1])
    fitparams_list = []
    error_list = []
    ymin = np.min(y)
    ymax = np.max(y)
    if ymax - ymin == 0:
        yrange = 1
    else:
        yrange = ymax - ymin
    ynorm = (y - ymin) / yrange

    def funcnorm(*args):
        return (func(*args) - ymin) / yrange

    def wr_funcnorm(pll):
        return np.sum((funcnorm(x, *pll) - ynorm) ** 2)

    for k, p0 in enumerate(p0list):
        p0 = np.ravel(p0)
        if method == "L-BFGS-B" or method == "TNC":
            fittedy = basinhopping(wr_funcnorm, p0, minimizer_kwargs={"method": method})
        else:
            fittedy = basinhopping(wr_funcnorm, p0, minimizer_kwargs={"method": method})
        error = fittedy.fun
        if printerrors == True:
            print(f"error = {error}")
        fitparams_list.append(fittedy.x)
        error_list.append(error)
    #     print(k / len(p0list), end="\r")
    # print("    ")
    best_error_idx = np.array(error_list).argmin()
    best_param = np.array(fitparams_list)[best_error_idx]
    if savetofile != False:
        with open(savetofile, "wb") as f:
            pickle.dump([error_list, fitparams_list], f)
    return [best_param, np.array(error_list)[best_error_idx]]


if __name__ == "__main__":

    def h(v, vhalf, k):
        return 1 / (1 + np.exp((v - vhalf) / -k))

    v = np.linspace(-0.100, 0.100, 3000)
    hinf = h(v, -0.050, -0.004)
    plt.plot(v, hinf, label="original")
    paramsfitted, errors = bruteforce(h, v, hinf, restrict=[[-1, -1], [1, 1]])
    for param in paramsfitted:
        plt.plot(v, h(v, *param), label="fitted")
    plt.legend()
    plt.show()

    plt.plot(v, hinf, label="original")
    paramfitted, error = scipy_fit(
        h, v, hinf, restrict=[[-1, -1], [1, 1]], p0list=paramsfitted
    )
    plt.plot(v, h(v, *paramfitted), label="fitted")
    plt.legend()
    plt.show()
