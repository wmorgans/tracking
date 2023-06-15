# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:08:11 2020

@author: j72687wm
"""

from blobDetectionAndTracking.scripts import *
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import scipy.stats as stats
from operator import itemgetter 
from itertools import groupby 
from distributions import TruncatedPowerLaw


def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
#%%

plot_width = 516/2

RW_synth = SyntheticData(mode = 'RW', nrows = 500, ncols = 500, npar = 50, tend = 12., dt = 0.012)
        
rnr_synth = SyntheticData(mode = 'runsAndRests', nrows = 500, ncols = 500, npar = 50, tend = 12., dt = 0.012)

#%%

#save stills
outDir = Path('../')

RW_synth.writeVid(outDir)
rnr_synth.writeVid(outDir)

plt.figure(figsize=set_size(plot_width))
plt.imshow(RW_synth.convim[:, :, 0], aspect = (5**.5 - 1) / 2)

#%%
RW_anal = TrackAnalysis(RW_synth.store_df)
RW_anal.plotAllTracks(RW_anal.df_tracks, 'Random Walk Synthetic Data')


rnr_anal = TrackAnalysis(rnr_synth.store_df)
rnr_anal.plotAllTracks(rnr_anal.df_tracks, '"Runs and Rests" Synthetic Data')

#%%

#plot jumscale and dist drawn from
beta = np.sqrt((3/pow(100/954.21,2))*2*0.012)
print(beta)

plt.figure(figsize=set_size(plot_width))
s = RW_synth.jumpscales 

count, bins, _ = plt.hist(s, 100, density=True)
fit = (1/beta)*np.exp(-(1/beta)*bins)
plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
plt.title('Stepsize distribution ' + 'D = ' + str(3) + ', dt = ' + str(0.012))
plt.xlim([0, 15])
plt.ylabel('$p(x)$')
plt.xlabel('$x$ $[pi]$')
plt.show()
#%%

#Plot intensities and dist drawn from

s = RW_synth.store_df.iloc[0:50, :]['int'].values
print(s)
mu = 140
sigma = 50

x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
fit = stats.norm.pdf(x, mu, sigma)
plt.figure(figsize=set_size(plot_width))
count, bins, _ = plt.hist(s, density=True)
plt.plot(x, max(count)*fit/max(fit), linewidth=2, color='r')
plt.title('Maximum intensity distribution ' + '$\mu$ = ' + str(140) + ', $\sigma$ = ' + str(50))
plt.ylabel('$p(i)$')
plt.xlabel('$i$ $[A.U]$')
plt.show()

#%%

#Plot velocity and dist drawn from
vel = rnr_synth.store_df['v'].dropna()

a = 1.34
b = 4.6
loc = 0
scale = 6 * 954/100

x = np.linspace(stats.beta.ppf(0.01, a, b, scale = scale),
                stats.beta.ppf(0.99, a, b, scale = scale), 100)

fit = stats.beta.pdf(x, a, b, scale = scale)

plt.figure(figsize=set_size(plot_width))
count, bins, _ = plt.hist(vel, density=True)
plt.plot(x,  max(count)*fit/max(fit), linewidth=2, color='r')
plt.title('Run velocity distribution')
plt.ylabel('$p(v)$')
plt.xlabel('$v$ $[pi/frame]$')
plt.show()


#%%

#Plot runs and rests and dist drawn from 

runDur = list(rnr_synth.durationRunsAndRests.values())
runDur = [item for sublist in runDur for item in sublist]
runDur = sorted(runDur, key=itemgetter(0))
#print(runDur)
groups = []
uniquekeys = []

for k, g in groupby(runDur, itemgetter(0)):
    groups.append(list(g))      # Store group iterator as a list
    uniquekeys.append(k)
#res = [[(i, j) for i, j in temp] for key, temp in groupby(runDur, key = itemgetter(0))] 

runDur = [i[1] for i in groups[0]]
restDur = [i[1] for i in groups[1]]

runDist = TruncatedPowerLaw(0.045, 1.352, 1.286, 10)
restDist = TruncatedPowerLaw(0.14, 0.518, 0.352, 10)


plt.figure()


count, binsRun, _ = plt.hist(runDur, density=True)
count, binsRest, _ = plt.hist(restDur, density=True)

plt.figure(figsize=set_size(plot_width))
logbinsRun = np.logspace(np.log10(binsRun[0]),np.log10(binsRun[-1]),len(binsRun))
plt.hist(runDur, density=True, bins = logbinsRun, color = 'r', alpha = 0.5, label = 'runs')
x = np.linspace(logbinsRun[0], 10, 1000)
fit = runDist.pdf(x)
plt.plot(x,  max(count)*fit/max(fit), linewidth=2, color='r')

logbinsRest = np.logspace(np.log10(binsRest[0]),np.log10(binsRest[-1]),len(binsRest))
plt.hist(restDur, density=True, bins = logbinsRest, color = 'b', alpha = 0.5, label = 'rests')
x = np.linspace(0, 10, 1000)
fit = restDist.pdf(x)
plt.plot(x,  max(count)*fit/max(fit), linewidth=2, color='b')


plt.title('Run duration distribution')
plt.yscale('log')
plt.xscale('log')
plt.xlim([logbinsRun[0], 10])
plt.ylabel('$p(t)$')
plt.xlabel('$t$ $[s]$')
plt.legend()
plt.show()