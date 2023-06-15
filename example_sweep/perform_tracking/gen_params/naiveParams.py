# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:12:41 2020

@author: j72687wm
"""
import itertools
import csv

npar = [50, 200]
diff_con = [1, 3, 5, 7, 9]
#mean_vel = [1, 3, 5, 7, 9, 11]

lookBack = [1, 2, 3, 5, 10]
maxCost = [10, 20, 40, 60, 80]

comb = [npar, diff_con, lookBack, maxCost]

allCombinations = list(itertools.product(*comb))

with open("naive_params.csv","w", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(allCombinations)
