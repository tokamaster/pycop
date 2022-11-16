from copulae import pseudo_obs
from pycop import empirical
from pyDOE import *
from scipy.integrate import dblquad
from scipy.stats.distributions import uniform
import numpy as np
from cmath import pi
import pandas as pd

def ishigami(x):
    return np.sin(x[0])+5*np.sin(x[1])**2+0.1*np.sin(x[0])*x[2]**4

lower_bounds = [-pi, -pi, -pi]
upper_bounds = [pi, pi, pi]

n_samples = 10000
samples = lhs(len(lower_bounds), samples=n_samples)

# Latin Hypercube Sampling of means
for i in range(len(lower_bounds)):
    samples[:, i] = uniform(loc=lower_bounds[i], scale=np.subtract(
        upper_bounds[i], lower_bounds[i])).ppf(samples[:, i])

results = []
for sample in samples:
    results.append(ishigami(sample))

data = {'x1':samples[:,0],'x2':samples[:,1],'x3':samples[:,2],'y':results}
pp = pd.DataFrame(data)
pp = pseudo_obs(pp)  # data has to be normalised ranked data for the Sklar's theorem to work: https://stats.stackexchange.com/questions/355085/when-modeling-a-copula-you-need-to-generate-pseudo-observations-why-what-is
copula1 = empirical(pp[['x1','y']])
copula2 = empirical(pp[['x2','y']])
copula3 = empirical(pp[['x3','y']])

copulae = [copula1,copula2,copula3]
indices = []

#for element in copulae:
#    f = lambda x,y: np.abs(element.cdf(x,y)-x*y)
#    integral = dblquad(f,0,1,0,1,epsabs=0.01)
#    indices.append(integral)
#print(indices)

from copulae import EmpiricalCopula

empcop = EmpiricalCopula(pp[['x1','x3','y']])

#f= lambda x,y,z: np.abs(empcop.cdf([x,y,z])-x*y*z)
#from scipy.integrate import tplquad
#print(tplquad(f,0,1,0,1,0,1,epsabs=0.01))

#b = 0
#for i in range(len(pp['x1'])):
#    a = np.abs(empcop.cdf([pp['x1'][i],pp['x2'][i],pp['y'][i]])-pp['x1'][i]*pp['x2'][i]*pp['y'][i])
#    b = b + a
#print(b)

import matplotlib.pyplot as plt
plt.scatter(pp['x1'],pp['x2'])
plt.savefig("x1x2.png")
plt.close()
plt.scatter(pp['x1'],pp['x3'])
plt.savefig("x1x3.png")
plt.close()
plt.scatter(pp['x1'],pp['y'])
plt.savefig("x1y.png")
plt.scatter(pp['x2'],pp['y'])
plt.savefig("x2y.png")
plt.scatter(pp['x3'],pp['y'])
plt.savefig("x3y.png")