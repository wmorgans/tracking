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
        print(data_comb)
        window = results
        i = 0
        for heading in data.keys():
            window = window.loc[(results[heading] == data_comb[i]), :]
            i += 1
        reduced = window.groupby(list(hyperparams.keys())).mean()
        
        try:
            best_params = reduced[selector].idxmax(axis=0, skipna=True)
        except(ValueError):
            print('missing values')
            continue
        #df.iloc[i] = reduced.loc[reduced['alpha'].idxmax(axis=0, skipna=True), :]
        #df.iloc[i][['method', 'data', 'lookBack', 'maxCost', 'nPar', 'diff_con']] = ['naive', 'rw', best_params[0], best_params[1], n_par, diff_con]
        print('data', data_comb)
        print('params', hyperparams.keys(), '\n' ,best_params)
        print('\n \n')
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
    for res_best in results_best:
        
        res_bymot_n = res_best.groupby(list(data.keys())).agg([np.mean, np.std])
        res_bymot_n = res_bymot_n.swaplevel(i = 0, j = 1)
        res_bymot_n = res_bymot_n.loc[200, metric]
        
        res_bymot.append(res_bymot_n)
     
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

fileLoc_n = Path('../../finalReport/rnr_naive_results_2.csv')
results_n = pd.read_csv(fileLoc_n)


fileLoc_kf = Path('../../finalReport/rnr_KF_results_2.csv')
results_kf = pd.read_csv(fileLoc_kf)

fileLoc_imm2 = Path('../../finalReport/rnr_imm2_results_3.csv')
results_imm2 = pd.read_csv(fileLoc_imm2)

fileLoc_imm3 = Path('../../finalReport/rnr_imm3_results_3.csv')
results_imm3 = pd.read_csv(fileLoc_imm3)


#Select Best HyperParams
data = {'mean_vel':[1, 3, 5, 7, 9, 11], 'nPar':[50, 200]}


#
hyper_n = {'lookBack':[1], 'maxCost':[40, 80, 120, 200]}
hyper_kf = {'lookBack':[1], 'maxCost':[60, 100, 200], 'sigAcc':[5, 15, 30, 50], 'sigMeas':[3, 5, 10]}
hyper_imm2 = {'lookBack':[1], 'maxCost':[50, 90, 190], 'sigPos':[10, 30, 90], 
              'sigAcc':[15, 40, 90],'sigMeas':[5, 10, 15]}
hyper_imm3 = {'maxCost':[50, 90, 190], 'sigPos':[10, 30, 90], 
              'sigAcc':[15, 40, 90],'sigMeas':[5, 10, 15]}

index = np.arange(0, 10)

df = pd.DataFrame(index=index, columns=results_n.columns)
comb = [data['nPar'], data['mean_vel']]
#comb2 = 
i = 0

print('naive')
results_best_n = getBestHyperParams(results_n, data, hyper_n, 'alpha')

print('kf')
results_best_kf = getBestHyperParams(results_kf, data, hyper_kf, 'alpha')

print('imm2')
results_best_imm2 = getBestHyperParams(results_imm2, data, hyper_imm2, 'alpha')

print('imm3')
results_best_imm3 = getBestHyperParams(results_imm3, data, hyper_imm3, 'alpha')

print(results_best_imm2.loc[(results_best_imm2['nPar'] == 200) &  (results_best_imm2['mean_vel'] == 9), 'alpha'])

#print(results_best.head(10))

titles = []
ylabs = []
met = []

titles.append(r"Plot of $\alpha$ for different velocities")
ylabs.append(r"$\alpha$")
met.append('alpha')

titles.append(r"Plot of $\beta$ for different velocities")
ylabs.append(r"$\beta$")
met.append('beta')

titles.append(r"Plot of $jac_{par}$ for different velocities")
ylabs.append("$jac_{par}$")
met.append('par_jaccard')

titles.append(r"Loyalty measure for different velocities")
ylabs.append(r"Ave. frames until $1^{st}$ dev.")
met.append('aveFramesOnFirstParticle')

titles.append(r"Loyalty measure for different velocities")
ylabs.append(r"Ave. frames until dev.")
met.append('propLoyalTracks')

xlab = r"$v$ $[\mu m/s]$ "  

results_best =  [results_best_n, results_best_kf, results_best_imm2, results_best_imm3]


for i in range(len(met)):
    plotLineOfMetrics(results_best, met[i], data, ['Naive', 'RW KF', 'IMM 2', 'IMM 3'], titles[i], xlab, ylabs[i])  

#print(df_std)    
method = ['naive', 'KF', 'IMM 2', 'IMM3']
    
for i in range(len(results_best)):
    toPlot = results_best[i].loc[:, ['nPar', 'mean_vel', 'alpha']].pivot_table(index = 'nPar', columns = 'mean_vel', values = 'alpha')
    
    title = r"Heat map of $\alpha$ for " + method[i] + " tracking"
    cbar = r"$\alpha$"
    xlab = r"$v$ $[\mu m/s]$ "
    
    plotHeatMap(toPlot, title, cbar, xlab)

        