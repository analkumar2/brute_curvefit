This is a simple script which tries to find the global minima using scipy.optimize.curve_fit as well as a parameter search over the parameter space.
It first generates ntol random models, then selects ntol*returnnfactor best models and does scipy.optimize.curve_fit on all of them. It then returns the best model of them all.
This script improves scipy.optimize.curve_fit in two ways - No need to give initial values and thus getting global minima instead of local minima. And second, it automatically normalize and standardizes the data.

FUTURE:
1. Parallelize
2. Use Genetic algorithm instead of brute-force

USAGE EXAMPLE:
import brute_curvefit as bf
import numpy as np
import matplotlib.pyplot as plt
def yourfunc(x, A,B,C):
	return (x-A)/B +C

params, error = bf.brute_then_scipy(func=yourfunc, x=np.arange(100), y=np.arange(100)**2, restrict=[[-1000,-1000,-1000],[1000,1000,1000]])
plt.plot(np.arange(100), np.arange(100)**2) #plotting data
plt.plot(np.arange(100), yourfunc(np.arange(100), *params)) #plotting the fitted function
plt.show()
#As simple as that