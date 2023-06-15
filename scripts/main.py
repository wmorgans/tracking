# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:15:43 2020

@author: j72687wm
"""
import utils
import blobDetection
import naiveTracker
import kalmanTracker
import immTracker
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import itertools
from pathlib import Path
import pickle as pkl

def analyseResults(readFile):
    #code to plot heatmaps of result df created below 
    results = pd.read_csv(readFile)
        
    for method in pd.unique(results['method']):
    
        toPlot = results.loc[results['method'] == method].pivot_table(index = 'nPar', columns = 'diff_con', values = 'beta',
                                                                         aggfunc = np.mean)
        
        
        plt.figure()
        plt.imshow(toPlot.values, cmap='hot', interpolation='nearest', aspect=toPlot.shape[1]/toPlot.shape[0], origin = 'lower')
        plt.title('Heat map of beta for ' + method)
        plt.ylabel('nPar')
        plt.xlabel('diffusion coefficient')
        plt.xticks(np.arange(0, 5, 1), [1, 3, 5, 7, 9], rotation = 45)
        plt.yticks(np.arange(0, 2, 1), [50, 200], rotation = 45)
        cbar = plt.colorbar()
        cbar.set_label('beta')
    

def createResultDF(readDir, writeFile):
    #code to create a results dataframe from pickled track dataframe and GT dataframe     
    trackDFs = {}
    f_names = []
    for f_name in filter(lambda fileInDir: fileInDir.endswith('.pkl'), os.listdir(readDir)):
            track_df = pd.read_pickle(fileDir + f_name)
            trackDFs[f_name[:f_name.rfind('.')]] = track_df
            f_names.append(f_name)
    
    f_names_ground = list(dict.fromkeys([f_name[:7] + '.csv' for f_name in f_names]))
    
    
    groundDFs = {}
    for f_name in f_names_ground:
        groundDFs[f_name[:f_name.rfind('.')]] = utils.loadCSV(fileDir + '\\RW\\' + f_name)
    
    nPars = ['10' ,'30', '50']
    charL_invs = ['0.25', '0.45', '0.65', '0.85']
    methods = ['naive', 'kalman', 'IMM']
    #results = pd.DataFrame(index=np.arange(len(f_names)),columns=['nPar', '1/CharL', 'method'], dtype = float)
    rows = []
    for nPar, charL_inv, method in itertools.product(nPars, charL_invs, methods):
        
        dfTrack = trackDFs[nPar + '_' + charL_inv + '_' + method]
        dfGround = groundDFs[nPar + '_' + charL_inv]
        
        tAn = TrackAnalysis(dfGround, dfTrack, 5)
        row = tAn.metrics
        row.update({'nPar': nPar, '1/CharL': charL_inv, 'method': method})
        rows.append(row)
    
    results = pd.DataFrame.from_dict(rows, orient='columns')
    
    results.to_pickle(writeFile)

    return
    
    
def main(fileDir, outDir):
    
    if True:
        
        '''
        Example flow:
            
            1) Get dataframe with data of all particles at each time step
                1.1) Load csv directly
                1.2) Blob detection from video
                1.3) Generate synthetic data
            2) Perform stitching resulting in dataframe with trajectories
                2.1) Naive tracker - stitching based on euclidean distance between tracks and detections
                2.2) Kalman Filter
                2.3) Interacting Multiple Model Filter
            3) Perform analysis
                1.1) No ground truth (i.e. no using syntheitc data) calc Mean Squared Displacement, 
                     histogram of track lengths, plot trajectories etc
                2.1) With Ground truth, correspondence and loyalty based metrics which access performance of tracking
        
        '''
        
        ################ STEP 1
        
        #Load csv directly
        #df_ground = utils.loadCSV(Path('../../runsAndRests_1_3.csv'))
        
        #Detect objects in video
        #src = utils.readVid(fileDir, singleFile = True)
        #utils.minMaxNorm(src, False, 255.0)
        #df_par = blobDetection.main(src)
        
        #Generate data
        testingSynth = SyntheticData(mode = 'runsAndRests', nrows = 400, ncols = 400, npar = 3, tend = 6., dt = 0.012)
        df_ground = testingSynth.store_df
        
        #testingSynth.df_store not currently in form required for stitching. Dave and load for correct form e.g.
        
        testingSynth.writeVid(Path('../example_vid_and_csv/'), 'testing')
        df_ground = utils.loadCSV(Path('../example_vid_and_csv/runsAndRests_3_testing.csv')) 
        
        ################ STEP 2
        lb = 1
        maxCost = 50
        
        #naive tracker
        niaveTracks  = naiveTracker.main(df_ground, len(pd.unique(df_ground['frame'])),
                                         lb = lb, maxDiff = maxCost)

        #kf tracker
        #kfTracks = kalmanTracker.main(df_ground, len(pd.unique(df_ground['frame']), lb, maxCost, F, P, Q, R, H, mode, x )
        
        #imm tracker
        #immTracks = immTracker.main(df_ground, len(pd.unique(df_ground['frame']), lb, maxCost, Fs, Ps, Qs, R, H, M, mu, modes, x)
        
        ################ STEP 3
        #anlysis with no ground truth tracks
        analysis = TrackAnalysis(niaveTracks)    
        
        analysis.plotAllTimeAve()    #MSD plot
        analysis.plotAllTracks(analysis.df_tracks, 'example of plotting trajectories')  #plot all trajectories (0 sub)
        analysis.plot_tracks(analysis.df_tracks, [1, 2])   #plot trajectory of trackID 1 and 2
        analysis.plotHistTrackLengths(analysis.df_tracks, 'exaple of track durations')  #plot hist of track duration 
        
        #anlysis with ground truth tracks. Automatically carries out correspondence and loyalty analysis.
        analysis_gt = TrackAnalysis(niaveTracks, df_ground)        
        
        analysis_gt.plotHistConsecutiveFrames('Frames until deviation of from particle')
        analysis_gt.plotHistConsecutiveFrames('Frames until deviation of from first particle', first = True)
        analysis_gt.plotLoyalty()   # Plot to visualise loyalty. GT object shown by colour, each row is a algorithm track.
        print(analysis.metrics)


        return
    
    if False:
        micronsToPix = 954.21/100 #[pi/um]
        testingSynth = SyntheticData(mode = 'RW', nrows = 400, ncols = 400, npar = 50, tend = 12., dt = 0.012)
        
        testingSynth.displayVid()
        testingSynth.writeVid(Path('../../'))
        df = utils.loadCSV(Path('../../RW_50_3.csv'))
        analysis = TrackAnalysis(df)
        analysis.plotAllTracks(analysis.df_tracks, 'RW tracks zero sub')
        
        plt.figure()
        plt.hist(pd.unique(testingSynth.store_df['v'])[~np.isnan(pd.unique(testingSynth.store_df['v']))] / micronsToPix)
        plt.show()
             
        runsAndRests = list(testingSynth.durationRunsAndRests.values())
        
        runsAndRests = [item for sublist in runsAndRests for item in sublist]
        runs = [item[1] for item in runsAndRests if item[0] == 'Dir']
        rests = [item[1] for item in runsAndRests if item[0] == 'RW']
        
        x = np.linspace(0, 20, 1000)
        pers = testingSynth.persistentTime.pdf(x)
        anti = testingSynth.antipersistentTime.pdf(x)
        
        print(runs)
        print(rests)
        
        plt.figure()
        plt.hist(runs, alpha=0.5, label='runs', color = 'r', density = True)
        plt.plot(x, pers, color = 'r')
        plt.hist(rests, alpha=0.5, label='rests', color = 'b', density = True)
        plt.plot(x, anti, color = 'b')
        plt.legend(loc='upper right')
        plt.show()
        
        
        return
    
#%%
if __name__ == '__main__':
    fileDir = Path("../../processedimages/13032020_lowNoise/")
    outDir = Path("../../processedimages/30032020_videoTesting/")
    main(fileDir, outDir)  