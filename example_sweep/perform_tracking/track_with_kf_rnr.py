import utils
import kalmanTrackerCV
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
from scipy.linalg import block_diag
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from filterpy.common import Q_discrete_white_noise

workingDir = os.getcwd()
#Load Args

params = [i for i in sys.argv[1].split(",")]
out_dir = Path(sys.argv[2])

#[npar, mean_vel, lookBack, maxCost, sigAcc, sigMeas]
npar = int(params[0])
mean_vel = int(params[1])
lookBack = int(params[2])
maxCost = int(params[3])
sigAcc = float(params[4])
sigMeas = float(params[5])


rows = []

mode = 'rnr'

#for measurement noise
xvar = yvar = sigMeas
avar = xvar*yvar

#As  acc and vel are /frame
dt = 1

#For const V
F = np.array([[1., dt, 0,  0, 0, 0],
				[0,  1,  0,  0, 0, 0],
				[0,  0, 1,  dt, 0, 0],
				[0,  0, 0,  1,  0, 0],
				[0,  0, 0,  0,  1, 0],
				[0,  0, 0,  0,  0, 1]])

P = np.array([[xvar, 0., 0,  0, 0, 0],
				[0,  100,  0,  0, 0, 0],
				[0,  0,  yvar, 0, 0, 0],
				[0,  0,   0, 100, 0, 0],
				[0,  0, 0, 0, sigMeas, 0],
				[0,  0, 0, 0, 0, avar]])

q =  Q_discrete_white_noise(2, dt = dt, var=sigAcc)
Q = block_diag(q, q, 1., 1.)


R = np.array([[sigMeas, 0., 0,  0],
			  [0,  sigMeas, 0,  0],
			  [0,  0,  sigMeas, 0],
			  [0,  0, 0,  avar]])
			  
H = np.array([[1., 0, 0, 0, 0, 0],
			  [0,  0, 1, 0, 0, 0],
			  [0,  0, 0, 0, 1, 0],
			  [0,  0, 0, 0, 0, 1]])
			  
x = ['x', 'y', 'i', 'a']

for i in range(10):
	fileLoc = Path('../rotation2/rnr_data/runsAndRests_' + str(npar) + "_" + str(mean_vel) + "_" + str(i) + ".csv")
	#read dataframe
	df_ground = utils.loadCSV(fileLoc)
	#trackCreation
	df_tracked = kalmanTrackerCV.main(df_ground.drop(['trackID'], axis = 1), len(pd.unique(df_ground['frame'])), 
	                               lookBack, maxCost,
	                               F, P, Q, R, H, mode, x) 
	#track analysis
	
	analysis = TrackAnalysis(df_tracked, df_ground)
	row = analysis.metrics
	row.update({'nPar': npar, 'mean_vel': mean_vel, 'method': 'KF', 'data': 'rnr', 'lookBack':lookBack, 'maxCost': maxCost, 'sigAcc': sigAcc, 'sigMeas':sigMeas})
	rows.append(row)
	
	del analysis
	del df_ground
	del df_tracked
	
results = pd.DataFrame.from_dict(rows, orient='columns')

if not os.path.isfile('rnr_KF_results_2.csv'):
   results.to_csv('rnr_KF_results_2.csv', header='column_names')
else: # else it exists so append without writing the header
   results.to_csv('rnr_KF_results_2.csv', mode='a', header=False)
