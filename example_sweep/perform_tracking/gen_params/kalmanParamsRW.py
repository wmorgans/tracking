# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:12:41 2020

@author: j72687wm
"""
import itertools
import csv


npar = [50, 200]
diff_con = [1, 3, 5, 7, 9]

maxCost = [10, 30, 60]
sigPos = [5, 15, 30, 50]
sigMeas = [3, 5, 10]

comb = [npar, diff_con, maxCost, sigPos, sigMeas]

allCombinations = list(itertools.product(*comb))

with open("RW_KF_params.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(allCombinations)
