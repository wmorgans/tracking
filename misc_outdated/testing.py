# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:01:02 2020

@author: j72687wm
"""
from pathlib import Path

from simulate_stills_modified import SyntheticData

vid = SyntheticData('RW', 400, 400, 100, 5, 0.012)
vid.displayVid()
vid.writeVid(Path('../'))



