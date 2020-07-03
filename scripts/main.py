# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:15:43 2020

@author: j72687wm
"""
import utils
import blobDetection
import trackCreation
from analysis import TrackAnalysis
from simulate_stills_modified import SyntheticData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import itertools
from pathlib import Path
import sys
sys.path.append(Path('./../externalScripts/itsample/'))
from externalScripts import itsample

def analyseResults(readFile):
    
    results = pd.read_pickle(readFile)
        
    for method in pd.unique(results['method']):
    
        alphaKal = results.loc[results['method'] == method].pivot(index = 'nPar', columns = '1/CharL', values = 'beta')
        
        
        plt.figure()
        plt.imshow(alphaKal.values, cmap='hot', interpolation='nearest', aspect=alphaKal.shape[1]/alphaKal.shape[0], origin = 'lower')
        plt.title('Heat map of alpha for ' + method)
        plt.ylabel('nPar')
        plt.xlabel('1/characteristic length')
        plt.xticks(np.arange(0, 4, 1), [0.25, 0.45, 0.65, 0.85], rotation = 45)
        plt.yticks(np.arange(0, 3, 1), [10, 30, 50], rotation = 45)
        cbar = plt.colorbar()
        cbar.set_label('beta')
    

def createResultDF(readDir, writeFile):
        
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
    
    
    if False:
        fileDir = Path("../../videos/simulated/05062020/")
        
        largeGT = utils.loadCSV(fileDir /'RW/50_0.85.csv')
        
        #tracks  = trackCreation.main(largeGT.drop(['trackID'], axis = 1), len(pd.unique(largeGT['frame'])))
        methods = ['naive', 'kalman', 'IMM']
        #easyTrack = pd.read_pickle(fileDir + '50_0.85_IMM.pkl')
        tracks = []
        for method in methods:
            fileName = '50_0.85_' + method + '.pkl'
            tracks.append(pd.read_pickle(fileDir /fileName))

        pixToMicrons = 954.21/100
        analysis = TrackAnalysis(tracks[0], pixsToMicrons = pixToMicrons)    

        analysisWithGround = TrackAnalysis(tracks[0], largeGT, pixToMicrons)        
        
        # analysis.plotAllTracks(analysis.df_ground, 'ground')
        # analysis.plotAllTracks(analysis.df_tracks, 'tracks')
        # analysis.plotLoyalty()
        # analysis.plotHistConsecutiveFrames('N. Consecutive Frames')
        
        analysis.plotEnsembleAve()
        analysisWithGround.plotEnsembleAve()
        
        analysis.plotAllTimeAve()
        analysisWithGround.plotAllTimeAve(ground = True)

        return
    
    if True:
        micronsToPix = 954.21/100 #[pi/um]
        testingSynth = SyntheticData(mode = 'runsAndRests', nrows = 400, ncols = 400, npar = 50, tend = 6., dt = 0.012)
        
        testingSynth.displayVid()
        testingSynth.writeVid(Path('../../'))
        df = utils.loadCSV(Path('../../runsAndRests_50_3.csv'))
        analysis = TrackAnalysis(df)
        analysis.plotAllTracks(analysis.df_tracks, 'testing')
        
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
    #Example flow
    fileDir = Path("../../processedimages/13032020_lowNoise/")
    outDir = Path("../../processedimages/200526_videoTesting/")
    
    #Load Video. 
    start = timer()
    srcs = utils.readVid(fileDir, singleFile = True)
    
    #All vids in file dir loaded. Select one
    src = [srcs[1]]
    
    #View video
    #utils.viewVideo(src, ['1'], ['False'])
    
    #
    src = [utils.minMaxNorm(vid, False, 255.0) for vid in src]
    
    #Detect vesicles in video. Function returns dataframe containing keypoints
    df_kp = blobDetection.main(src[0])
    
    ##OR load csv of synthetic data directly and skip blob detection.
    #groundTruth_df = utils.loadCSV(fileDir + filename)
    #df_tracks = groundTruth_df.drop['trackID']
   
    #Create tracks. Will automatically run naive, kalman and IMM methods. Each will have a seperate dataframe
    df_tracks = trackCreation.main(df_kp, len(pd.unique(df_kp['frame'])))
    #print(type(imWithTracks[0]))
    
    end = timer()
    
    print("total time", str(end - start))
    
    #Now do what you want e.g. save video, draw tracks on video etc. 
    #utils.viewVideo(imWithTracks, ['reg', 'kalman', 'IMM'], [False, False, False])
    #utils.writeVid(imWithTracks, outDir, '19_05', 'MJPG', 12, True)
    #utils.writeVid(imWithTracks_kalman, outDir, '19_05_k', 'MJPG', 12, True)
    
    
  
    
    
    
    

#%%
if __name__ == '__main__':
    fileDir = Path("../../processedimages/13032020_lowNoise/")
    outDir = Path("../../processedimages/30032020_videoTesting/")
    main(fileDir, outDir)  