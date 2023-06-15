import utils
import naiveTracker
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

workingDir = os.getcwd()
#Load Args

params = [i for i in sys.argv[1].split(",")]
out_dir = Path(sys.argv[2])

#comb = [npar, mean_vel, lookBack, maxCost]
npar = params[0]
mean_vel = params[1]
lookBack = int(params[2])
maxCost = int(params[3])

rows = []
for i in range(10):
	fileLoc = Path('../rotation2/rnr_data/runsAndRests_' + npar + "_" + mean_vel + "_" + str(i) + ".csv")
	#read dataframe
	df_ground = utils.loadCSV(fileLoc)
	#trackCreation
	df_tracked = naiveTracker.main(df_ground.drop(['trackID'], axis = 1), len(pd.unique(df_ground['frame'])), 
	                               lb = lookBack, maxDiff = maxCost) 
	#track analysis
	
	analysis = TrackAnalysis(df_tracked, df_ground)
	row = analysis.metrics
	row.update({'nPar': npar, 'mean_vel': mean_vel, 'method': 'naive', 'data': 'rnr', 'lookBack':lookBack, 'maxCost': maxCost})
	rows.append(row)
	
	del analysis
	del df_ground
	del df_tracked
	
results = pd.DataFrame.from_dict(rows, orient='columns')

if not os.path.isfile('rnr_naive_results_2.csv'):
   results.to_csv('rnr_naive_results_2.csv', header='column_names')
else: # else it exists so append without writing the header
   results.to_csv('rnr_naive_results_2.csv', mode='a', header=False)
