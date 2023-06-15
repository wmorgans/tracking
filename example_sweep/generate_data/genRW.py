import utils
import trackCreation
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

workingDir = os.getcwd()
#Load Args

params = [int(i) for i in sys.argv[1].split(",")]
out_dir = Path(sys.argv[2])


for i in range(10):

	testingSynth = SyntheticData(mode = 'RW', nrows = 500, ncols = 500, \
										 npar = params[0], tend = 12., dt = 0.012,\
										 diff_coeff_um = params[1])
	
	testingSynth.writeVid(out_dir, i)
	print(i)
	del testingSynth
