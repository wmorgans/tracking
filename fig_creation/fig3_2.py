# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2

@author: j72687wm
"""

import utils
import immTracker
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
from filterpy.common import Q_discrete_white_noise
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import block_diag
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)

cost = 90
sigPos = 10
sigAcc = 15
sigMeas = 15

dt = 1

M = np.array([[0.7, 0.15, 0.15],
              [0.2, 0.7, 0.1],
              [0.25, 0.25, 0.5]])
mu = [0.9, 0.1, 0]

modes = ['RW', 'CV', 'CA']
    
    #For RW
F_r = np.array([[1., 0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_r = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  0, 0,     0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, sigMeas, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

Q_r = np.array([[sigPos, 0., 0,  0, 0, 0, 0, 0],
				[0,  0, 0, 0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigPos, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1., 0],
				[0,  0, 0,  0, 0, 0, 0, 1.]])

#For const V
F_v = np.array([[1., dt, 0,  0, 0, 0, 0, 0],
				[0,  1, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, dt, 0, 0, 0],
				[0,  0, 0,  0, 1, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_v = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  100, 0,    0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,   100, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0,    0, sigMeas, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

q_v =  Q_discrete_white_noise(2, dt = dt, var=sigAcc)
Q_v = block_diag(q_v, 0., q_v, 0., 1, 1.)

#For const A
F_a = np.array([[1., dt, 0.5*(dt**2),  0, 0, 0, 0, 0],
				[0,  1, dt,  0, 0, 0, 0, 0],
				[0,  0, 1,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, dt, 0.5*(dt**2), 0, 0],
				[0,  0, 0,  0, 1, dt, 0, 0],
				[0,  0, 0,  0, 0, 1, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_a = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  100, 0,    0, 0, 0, 0, 0],
				[0,  0, 100,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,   100, 0, 0, 0],
				[0,  0, 0,  0,    0, 100, 0, 0],
				[0, 0, 0, 0, 0, 0, sigMeas, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

q_a = Q_discrete_white_noise(3, dt = dt, var=sigAcc)
Q_a = block_diag(q_a, q_a, 1., 1.)





R = np.array([[sigMeas, 0., 0,  0],
			  [0,  sigMeas, 0,  0],
			  [0,  0,  sigMeas, 0],
			  [0,  0, 0,  sigMeas*sigMeas]])
			  
H = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
			  [0,  0, 0, 1, 0, 0, 0, 0],
			  [0,  0, 0, 0, 0, 0, 1, 0],
			  [0,  0, 0, 0, 0, 0, 0, 1]])
			  
x = ['x', 'y', 'i', 'a']

Fs = [F_r, F_v, F_a]
Ps = [P_r, P_v, P_a]
Qs = [Q_r, Q_v, Q_a]


#velDistNorm = {'type':'gaus', 'mean': 3, 'sigma': 0.5}


# testingSynth_3 = SyntheticData(mode = 'runsAndRests', nrows = 500, ncols = 500, npar = 200, tend = 12., dt = 0.012,
#                              velDist = velDistNorm)

# testingSynth_3.writeVid(Path('../../'), '')
ground_3 = utils.loadCSV(Path('../../runsAndRests_200_3_.csv'))

print(len(pd.unique(ground_3['frame'])))
# track_3  = immTracker.main(ground_3.drop(['trackID'], axis = 1), len(pd.unique(ground_3['frame'])),
#                               1, cost, Fs, Ps, Qs, R, H, M, mu,modes, x)

# track_3.to_pickle(Path('../../vel3.pkl'))
track_3 = pd.read_pickle(Path('../../vel3.pkl'))

pixToMicrons = 100/954.21
analysis_3 = TrackAnalysis(track_3, ground_3, pixToMicrons = pixToMicrons)
analysis_3.plotLoyalty()
analysis_3.plotAllTimeAve()
analysis_3.plotAllTimeAve(ground=True)

print(analysis_3.metrics)

cost = 190
sigPos = 10
sigAcc = 90
sigMeas = 15

F_r = np.array([[1., 0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_r = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  0, 0,     0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, sigMeas, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

Q_r = np.array([[sigPos, 0., 0,  0, 0, 0, 0, 0],
				[0,  0, 0, 0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigPos, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1., 0],
				[0,  0, 0,  0, 0, 0, 0, 1.]])

#For const V
F_v = np.array([[1., dt, 0,  0, 0, 0, 0, 0],
				[0,  1, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, dt, 0, 0, 0],
				[0,  0, 0,  0, 1, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_v = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  100, 0,    0, 0, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,   100, 0, 0, 0],
				[0,  0, 0,  0,    0, 0, 0, 0],
				[0,  0, 0,  0,    0, sigMeas, 0, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

q_v =  Q_discrete_white_noise(2, dt = dt, var=sigAcc)
Q_v = block_diag(q_v, 0., q_v, 0., 1, 1.)

#For const A
F_a = np.array([[1., dt, 0.5*(dt**2),  0, 0, 0, 0, 0],
				[0,  1, dt,  0, 0, 0, 0, 0],
				[0,  0, 1,  0, 0, 0, 0, 0],
				[0,  0, 0,  1, dt, 0.5*(dt**2), 0, 0],
				[0,  0, 0,  0, 1, dt, 0, 0],
				[0,  0, 0,  0, 0, 1, 0, 0],
				[0,  0, 0,  0, 0, 0, 1, 0],
				[0,  0, 0,  0, 0, 0, 0, 1]])

P_a = np.array([[sigMeas, 0., 0,  0, 0, 0, 0, 0],
				[0,  100, 0,    0, 0, 0, 0, 0],
				[0,  0, 100,  0,    0, 0, 0, 0],
				[0,  0, 0,  sigMeas, 0, 0, 0, 0],
				[0,  0, 0,  0,   100, 0, 0, 0],
				[0,  0, 0,  0,    0, 100, 0, 0],
				[0, 0, 0, 0, 0, 0, sigMeas, 0],
				[0,  0, 0,  0, 0, 0, 0, sigMeas*sigMeas]])

q_a = Q_discrete_white_noise(3, dt = dt, var=sigAcc)
Q_a = block_diag(q_a, q_a, 1., 1.)





R = np.array([[sigMeas, 0., 0,  0],
			  [0,  sigMeas, 0,  0],
			  [0,  0,  sigMeas, 0],
			  [0,  0, 0,  sigMeas*sigMeas]])
			  
H = np.array([[1., 0, 0, 0, 0, 0, 0, 0],
			  [0,  0, 0, 1, 0, 0, 0, 0],
			  [0,  0, 0, 0, 0, 0, 1, 0],
			  [0,  0, 0, 0, 0, 0, 0, 1]])
			  
x = ['x', 'y', 'i', 'a']

Fs = [F_r, F_v, F_a]
Ps = [P_r, P_v, P_a]
Qs = [Q_r, Q_v, Q_a]

velDistNorm = {'type':'gaus', 'mean': 11, 'sigma': 0.5}


# testingSynth_11 = SyntheticData(mode = 'runsAndRests', nrows = 500, ncols = 500, npar = 200, tend = 12., dt = 0.012,
#                              velDist = velDistNorm)

# testingSynth_11.writeVid(Path('../../'), '')
ground_11 = utils.loadCSV(Path('../../runsAndRests_200_11_.csv'))

# track_11 = immTracker.main(ground_11.drop(['trackID'], axis = 1), len(pd.unique(ground_11['frame'])),
#                               1, cost, Fs, Ps, Qs, R, H, M, mu, modes, x)

# track_11.to_pickle(Path('../../vel11.pkl'))
track_11 = pd.read_pickle(Path('../../vel11.pkl'))

analysis_11 = TrackAnalysis(track_11, ground_11, pixToMicrons = pixToMicrons)
analysis_11.plotLoyalty() 
analysis_11.plotAllTimeAve()
analysis_11.plotAllTimeAve(ground=True)   

print(analysis_11.metrics)