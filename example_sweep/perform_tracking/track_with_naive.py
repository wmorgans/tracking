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

#params [npar, diff_con, lookBack, maxCost]
npar = params[0]
diff_con = params[1]
lookBack = int(params[2])
maxCost = int(params[3])

rows = []
for i in range(10):
	fileLoc = Path('../rotation2/RW_data/RW_' + npar + "_" + diff_con + "_" + str(i) + ".csv")
	#read dataframe
	df_ground = utils.loadCSV(fileLoc)
	#trackCreation
	df_tracked = naiveTracker.main(df_ground.drop(['trackID'], axis = 1), len(pd.unique(df_ground['frame'])), 
	                               lb = lookBack, maxDiff = maxCost) 
	#track analysis
	
	analysis = TrackAnalysis(df_tracked, df_ground)
	row = analysis.metrics
	row.update({'nPar': npar, 'diff_con': diff_con, 'method': 'naive', 'data': 'RW', 'lookBack':lookBack, 'maxCost': maxCost})
	rows.append(row)
	
	del analysis
	del df_ground
	del df_tracked
	
results = pd.DataFrame.from_dict(rows, orient='columns')

if not os.path.isfile('RW_naive_results.csv'):
   results.to_csv('RW_naive_results.csv', header='column_names')
else: # else it exists so append without writing the header
   results.to_csv('RW_naive_results.csv', mode='a', header=False)
