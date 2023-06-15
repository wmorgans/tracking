# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:02:59 2020

@author: j72687wm
"""
# RW tracking


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
import utils
import matplotlib

def double_std(array):
    return np.std(array) * 2

def getBestHyperParams(results, data, hyperparams, selector):
    i = 0
    results_best = pd.DataFrame(columns=results.columns)
    
    comb = list(data.values())

    for data_comb in itertools.product(*comb):
        
        window = results
        i = 0
        for heading in data.keys():
            window = window.loc[(results[heading] == data_comb[i]), :]
            i += 1
        reduced = window.groupby(list(hyperparams.keys())).mean()
         
        best_params = reduced[selector].idxmax(axis=0, skipna=True)
        #df.iloc[i] = reduced.loc[reduced['alpha'].idxmax(axis=0, skipna=True), :]
        #df.iloc[i][['method', 'data', 'lookBack', 'maxCost', 'nPar', 'diff_con']] = ['naive', 'rw', best_params[0], best_params[1], n_par, diff_con]
        print('data', data_comb)
        print('params', best_params)
        #print(reduced_std.loc[best_params, :])
        #print(window.loc[(window['lookBack'] == best_params[0]) & (window['maxCost'] == best_params[1]), : ])
        
        window_to_add = window
        i = 0
        for heading in list(hyperparams.keys()):
            window_to_add = window_to_add.loc[window_to_add[heading] == best_params[i]]
            i += 1
        results_best = results_best.append(window_to_add)
        i += 1
    
    return results_best

def plotHeatMap(toPlot, title, cbar_lab, xlab):
    
    plt.figure(figsize=utils.set_size(3*516/4))
    plt.imshow(toPlot.values.astype('float'), cmap='hot', interpolation='nearest', aspect=1/((5**.5 - 1) / 2), origin = 'lower')
    plt.title(title)
    plt.ylabel("Number of Particles")
    plt.xlabel(xlab)
    plt.xticks(np.arange(0, 5, 1), [1, 3, 5, 7, 9])
    plt.yticks(np.arange(0, 2, 1), [50, 200])
    cbar = plt.colorbar()
    cbar.set_label(cbar_lab)
    for (j,i),label in np.ndenumerate(toPlot.values):
        plt.text(i,j,np.round(label, 3),ha='center',va='center')

def plotLineOfMetrics(results_best, metric, data, methods, title, xlab, ylab):
    res_bymot = []
    print(metric)
    for res_best in results_best:
        
        res_bymot_n = res_best.groupby(list(data.keys())).agg([np.mean, np.std])
        res_bymot_n = res_bymot_n.swaplevel(i = 0, j = 1)
        res_bymot_n = res_bymot_n.loc[200, metric]
        
        res_bymot.append(res_bymot_n)
    
    print(res_bymot[0])
    ax1 = res_bymot[0].plot(kind = "line",  y = "mean", 
                   title = title, yerr = "std", figsize = utils.set_size(3*516/4))
    for toPlot in res_bymot[1:]: 
        toPlot.plot(ax = ax1, kind = "line",  y = "mean", yerr = "std")
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)    
    ax1.legend(methods)
    
    
#start script    
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)
    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

fileLoc_n = Path('../../finalReport/RW_naive_results.csv')
results_n = pd.read_csv(fileLoc_n)


fileLoc_kf = Path('../../finalReport/RW_KF_results.csv')
results_kf = pd.read_csv(fileLoc_kf)


#Select Best HyperParams
data = {'diff_con':[1, 3, 5, 7, 9], 'nPar':[50, 200]}


#
hyper_n = {'lookBack':[1, 2, 3, 5, 10], 'maxCost':[10, 20, 40, 60, 80]}
hyper_kf = {'maxCost':[10, 30, 60], 'sigP':[5, 15, 30, 50], 'sigMeas':[3, 5, 10]}

index = np.arange(0, 10)

df = pd.DataFrame(index=index, columns=results_n.columns)
comb = [data['nPar'], data['diff_con']]
#comb2 = 
i = 0

print('naive')
results_best_n = getBestHyperParams(results_n, data, hyper_n, 'alpha')

print('kf')
results_best_kf = getBestHyperParams(results_kf, data, hyper_kf, 'alpha')

#print(results_best.head(10))

titles = []
ylabs = []
met = []

titles.append(r"Plot of $\alpha$ for different diffusion coefficients")
ylabs.append(r"$\alpha$")
met.append('alpha')

titles.append(r"Plot of $\beta$ for different diffusion coefficients")
ylabs.append(r"$\beta$")
met.append('beta')

titles.append(r"Plot of $jac_{par}$ for different diffusion coefficients")
ylabs.append("$jac_{par}$")
met.append('par_jaccard')

titles.append(r"Loyalty measure for different diffusion coefficients")
ylabs.append(r"Ave. frames until $1^{st}$ dev.")
met.append('propLoyalTracks')

xlab = r"$D$ $[\mu m^2/s]$ "

results_best =  [results_best_n, results_best_kf]


for i in range(len(met)):
    plotLineOfMetrics(results_best, met[i], data, ['Naive', 'RW kalman Filter'], titles[i], xlab, ylabs[i])  

#print(df_std)    
method = ['naive', 'KF']
    
for i in range(len(results_best)):
    toPlot = results_best[i].loc[:, ['nPar', 'diff_con', 'alpha']].pivot_table(index = 'nPar', columns = 'diff_con', values = 'alpha')
    
    title = r"Heat map of $\alpha$ for " + method[i] + " tracking"
    cbar = r"$\alpha$"
    xlab = r"$D$ $[\mu m^2/s]$ "
    
    plotHeatMap(toPlot, title, cbar, xlab)

        