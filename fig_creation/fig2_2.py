# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:38:15 2020

@author: j72687wm
"""

import utils
import blobDetection
import naiveTracker
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import itertools
from pathlib import Path
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 11}

matplotlib.rc('font', **font)

# testingSynth_1 = SyntheticData(mode = 'RW', nrows = 500, ncols = 500, npar = 50, tend = 12., dt = 0.012,
#                              diff_coeff_um = 1)

# testingSynth_1.writeVid(Path('../../'), '')
ground_1 = utils.loadCSV(Path('../../RW_50_1_.csv'))

track_1  = naiveTracker.main(ground_1.drop(['trackID'], axis = 1), 
                             len(pd.unique(ground_1['frame'])), lb = 1, maxDiff = 20)


pixToMicrons = 100/954.21
analysis_1 = TrackAnalysis(track_1, ground_1, pixToMicrons = pixToMicrons)
analysis_1.plotLoyalty()
analysis_1.plotAllTimeAve()
#analysis_1.plotAllTimeAve(ground=True)    


# testingSynth_9 = SyntheticData(mode = 'RW', nrows = 500, ncols = 500, npar = 50, tend = 12., dt = 0.012,
#                              diff_coeff_um = 9)

# testingSynth_9.writeVid(Path('../../'), '')
ground_9 = utils.loadCSV(Path('../../RW_50_9_.csv'))

track_9 = naiveTracker.main(ground_9.drop(['trackID'], axis = 1),
                            len(pd.unique(ground_9['frame'])), lb = 1, maxDiff = 80)

analysis_9 = TrackAnalysis(track_9, ground_9, pixToMicrons = pixToMicrons)
analysis_9.plotLoyalty() 
analysis_9.plotAllTimeAve()
#analysis_9.plotAllTimeAve(ground=True)   