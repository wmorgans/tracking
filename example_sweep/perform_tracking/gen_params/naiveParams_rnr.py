# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:12:41 2020

@author: j72687wm
"""
import itertools
import csv

npar = [50, 200]
#diff_con = [1, 3, 5, 7, 9]
mean_vel = [1, 3, 5, 7, 9, 11]

lookBack = [1]
maxCost = [40, 80, 120, 200]

comb = [npar, mean_vel, lookBack, maxCost]

allCombinations = list(itertools.product(*comb))

with open("naive_rnr_params.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(allCombinations)
