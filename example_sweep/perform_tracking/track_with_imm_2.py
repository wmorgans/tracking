import utils
import immTracker2
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

#[npar, mean_vel, lookBack, maxCost, sigPos, sigAcc, sigMeas]
npar = int(params[0])
mean_vel = int(params[1])
lookBack = int(params[2])
maxCost = int(params[3])
sigPos = float(params[4])
sigAcc = float(params[5])
sigMeas = float(params[6])

print(lookBack)
rows = []

modes = ['CV', 'RW']

#for measurement noise
xvar = yvar = sigMeas
avar = xvar*yvar

#As  acc and vel are /frame
dt = 1

M = np.array([[0.7, 0.3],
			  [0.3, 0.7]])
mu = [0.9, 0.1]

#For const V
F_v = np.array([[1., dt, 0,  0, 0, 0],
				[0,  1,  0,  0, 0, 0],
				[0,  0, 1,  dt, 0, 0],
				[0,  0, 0,  1,  0, 0],
				[0,  0, 0,  0,  1, 0],
				[0,  0, 0,  0,  0, 1]])
				
F_r = np.array([[1., 0, 0,  0,  0, 0],
				[0,  0, 0,  0,  0, 0],
				[0,  0, 1,  0,  0, 0],
				[0,  0, 0,  0,  0, 0],
				[0,  0, 0,  0,  1, 0],
				[0,  0, 0,  0,  0, 1]])

P_v = np.array([[xvar, 0., 0,  0, 0, 0],
				[0,  100,  0,  0, 0, 0],
				[0,  0,  yvar, 0, 0, 0],
				[0,  0,   0, 100, 0, 0],
				[0,  0, 0, 0, sigMeas, 0],
				[0,  0, 0, 0, 0, avar]])
				
P_r = np.array([[xvar, 0., 0,  0, 0, 0],
				[0,  0,  0,  0, 0, 0],
				[0,  0,  yvar, 0, 0, 0],
				[0,  0,   0, 0, 0, 0],
				[0,  0, 0, 0, sigMeas, 0],
				[0,  0, 0, 0, 0, avar]])

q =  Q_discrete_white_noise(2, dt = dt, var=sigAcc)
Q_v = block_diag(q, q, 1., 1.)

Q_r = np.array([[sigPos, 0., 0,  0, 0, 0],
				[0,  0, 0, 0, 0, 0],
				[0,  0, sigPos,  0, 0, 0],
				[0,  0, 0,  0, 0, 0],
				[0,  0, 0,  0, 1, 0],
				[0,  0, 0,  0, 0, 1.]]) 


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
	df_tracked = immTracker2.main(df_ground.drop(['trackID'], axis = 1), len(pd.unique(df_ground['frame'])), 
	                               lookBack, maxCost,
	                               [F_v, F_r], [P_v, P_r], [Q_v, Q_r], R, H, M, mu, modes, x) 
	#track analysis
	
	analysis = TrackAnalysis(df_tracked, df_ground)
	row = analysis.metrics
	row.update({'nPar': npar, 'mean_vel': mean_vel, 'method': 'imm2', 'data': 'rnr', 'lookBack':lookBack, 'maxCost': maxCost, 'sigAcc': sigAcc, 'sigMeas':sigMeas, 'sigPos': sigPos})
	rows.append(row)
	
	del analysis
	del df_ground
	del df_tracked
	
results = pd.DataFrame.from_dict(rows, orient='columns')

if not os.path.isfile('rnr_imm2_results_3.csv'):
   results.to_csv('rnr_imm2_results_3.csv', header='column_names')
else: # else it exists so append without writing the header
   results.to_csv('rnr_imm2_results_3.csv', mode='a', header=False)
