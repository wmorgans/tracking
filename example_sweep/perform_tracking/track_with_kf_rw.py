import utils
import kalmanTrackerRW
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

#[npar, diff_con, lookBack, maxCost, sigPos, sigMeas]
npar = params[0]
diff_con = params[1]
maxCost = int(params[2])
sigPos = float(params[3])
sigMeas = float(params[4])
lookBack = 1

rows = []

mode = 'RW'

#for measurement noise
xvar = yvar = sigMeas
avar = xvar*yvar

#for process noise
sigP = sigPos #pi/frame

F = np.array([[1., 0, 0, 0],
			  [0,  1, 0, 0],
			  [0,  0, 1, 0],
			  [0,  0, 0, 1]])

P = np.array([[sigMeas, 0., 0,  0],
			  [0,  sigMeas, 0,  0],
			  [0,  0,  sigMeas, 0],
			  [0,  0, 0,  avar]])

Q = np.array([[sigP, 0., 0,  0],
			  [0,  sigP, 0,  0],
			  [0,  0,    1,  0],
			  [0,  0,    0,  1]])

R = np.array([[sigMeas, 0., 0,  0],
			  [0,  sigMeas, 0,  0],
			  [0,  0,  sigMeas, 0],
			  [0,  0, 0,  avar]])
			  
H = np.array([[1., 0, 0, 0],
			  [0,  1, 0, 0],
			  [0,  0, 1, 0],
			  [0,  0, 0, 1]])
			  
x = ['x', 'y', 'i', 'a']

for i in range(10):
	fileLoc = Path('../rotation2/RW_data/RW_' + npar + "_" + diff_con + "_" + str(i) + ".csv")
	#read dataframe
	df_ground = utils.loadCSV(fileLoc)
	#trackCreation
	df_tracked = kalmanTrackerRW.main(df_ground.drop(['trackID'], axis = 1), len(pd.unique(df_ground['frame'])), 
	                               lookBack, maxCost,
	                               F, P, Q, R, H, mode, x) 
	#track analysis
	
	analysis = TrackAnalysis(df_tracked, df_ground)
	row = analysis.metrics
	row.update({'nPar': npar, 'diff_con': diff_con, 'method': 'KF', 'data': 'RW', 'lookBack':lookBack, 'maxCost': maxCost, 'sigP': sigPos, 'sigMeas':sigMeas})
	rows.append(row)
	
	del analysis
	del df_ground
	del df_tracked
	
results = pd.DataFrame.from_dict(rows, orient='columns')

if not os.path.isfile('RW_KF_results.csv'):
   results.to_csv('RW_KF_results.csv', header='column_names')
else: # else it exists so append without writing the header
   results.to_csv('RW_KF_results.csv', mode='a', header=False)
