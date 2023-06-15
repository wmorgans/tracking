# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:12:41 2020

@author: j72687wm
"""
import itertools
import csv


npar = [50, 200]
mean_vel = [1, 3, 5, 7, 9, 11]

maxCost = [50, 90, 190]
sigPos = [10, 30, 90]
sigAcc = [15, 40, 90]
sigMeas = [5, 10, 15]

comb = [npar, mean_vel, maxCost, sigPos, sigAcc, sigMeas]

allCombinations = list(itertools.product(*comb))

with open("IMM3_params.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(allCombinations)
