# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:12:41 2020

@author: j72687wm
"""
import itertools
import csv

npar = [50, 200]
mean_vel = [1, 3, 5, 7, 9, 11]

maxCost = [60, 100, 200]
sigAcc = [5, 15, 30, 50]
sigMeas = [3, 5, 10]
lookBack = [1]

comb = [npar, mean_vel, lookBack, maxCost, sigAcc, sigMeas]

allCombinations = list(itertools.product(*comb))

with open("CV_KF_params.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(allCombinations)
