# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:38:45 2020

@author: j72687wm
"""

from distributions import TruncatedPowerLaw
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

#What should the upper limit be??
x_max = 20

x_per = np.linspace(0.045, x_max, 10000)
x_anti = np.linspace(0.14, x_max, 10000)

persistentDist = TruncatedPowerLaw(mu = 1.352, tau_0 = 0.045, lamda =1.286, name = 'persistentDist', max_t = x_max)
pdf_per = persistentDist.pdf(x = x_per)

antipersistentDist = TruncatedPowerLaw(mu = 0.518, tau_0 = 0.14, lamda = 0.352,name = 'antipersistentDist', max_t = x_max)
pdf_anti = antipersistentDist.pdf(x = x_anti)

#Rab5
#anti persistant
#μ=0.518±0.004 , τ0=0.140±0.002s, λ=0.352±0.002s−1

#persistant
#μ=1.352±0.102, τ0=0.045±0.006s,  λ=1.286±0.142

print('anti persistent', simps(pdf_anti, x_anti))
print('persistent', simps(pdf_per, x_per))

per_samples = persistentDist.rvs(size = 1000)
anti_samples = antipersistentDist.rvs(size = 1000)

per_logbins = np.logspace(np.log(np.amin(per_samples)),np.log(np.amax(per_samples)),100)
anti_logbins = np.logspace(np.log(np.amin(anti_samples)),np.log(np.amax(anti_samples)),100)

print('per')
print(np.min(per_samples), np.max(per_samples))

print('anti')
print(np.min(anti_samples), np.max(anti_samples))

plt.figure()
plt.hist(anti_samples, bins = anti_logbins,  color='r', alpha = 0.5, density = True)
plt.loglog(x_anti, pdf_anti, linewidth=2, color='r')
plt.title('anti-persistent')
plt.show()

plt.figure()
plt.hist(per_samples, bins = per_logbins, color='b', alpha = 0.5, density = True)
plt.loglog(x_per, pdf_per, linewidth = 2, color='b')
plt.title('persistent')
plt.show()

from scipy.stats import alpha, norm, beta
from scipy.optimize import curve_fit



fig, ax = plt.subplots(1, 1)

n = 2
data =  [0.1]*n + [0.5]*2*n + [0.65]*5*n + [0.9]*6*n + [1.1]*4*n + [1.4]*4*n + [1.6]*3*n + [2.5]*2 + [3.5] + [5.9]

a, b, loc, scale = beta.fit(data, floc = 0, fscale = 6)
r = beta.rvs(a, b, loc = loc, scale = scale, size=10000)
print('a', 'b', 'loc', 'scale')
print(a, b, loc, scale)

print(beta.ppf(0.001, a, b, scale = scale, loc = loc))
print(beta.ppf(0.99, a, b, scale = scale, loc = loc))

x = np.linspace(0, beta.ppf(0.99, a, b, scale = scale, loc = loc), 1000)

ax.plot(x, beta.pdf(x, a, b, loc = loc, scale = scale),
       'r-', lw=5, alpha=0.6, label='gamma pdf')
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2, bins = 100)
print(beta.pdf(6, a, b, loc = loc, scale = scale))
