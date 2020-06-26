# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:10:41 2020

@author: j72687wm
"""
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib
import sys
import itertools
import psutil
import seaborn as sns
from pathlib import Path

class CorrespondanceAnalysis:
    
    
    def __init__(self, dF_ground, dF_tracks, thresh = 5, **kw):
        self.df_ground = dF_ground.copy()
        self.df_tracks = dF_tracks.copy()
        
        self.n_gtTracks = len(pd.unique(self.df_ground.loc[:, 'trackID']))
        self.n_dTracks = len(pd.unique(self.df_tracks.loc[:, 'trackID']))
        
        
        self.nFrames = len(pd.unique(self.df_ground.loc[:, 'timePoint']))
        self.thresh = thresh
        
        self.mat_ground = self.__dfToArray(self.df_ground)
        self.mat_tracks = self.__dfToArray(self.df_tracks)
        self.metrics = {}
        
        self.__pairGTandAlgTracks()            #Construct costmatrix then create optimal set of pairs
        self.__calcAlphaAndBeta()              #calc alpha and beta
        self.__particleAssociationMetrics()    #calc tp, fn, fp and jaccard for particles
        self.__trackAssociationMetrics()       #calc tp, fn, fp and jaccard for tracks
        self.__localisationMetrics()           #calc RMSE, min, max and SD for tp particles
        
        super().__init__(**kw)
        
    def __pairGTandAlgTracks(self):
        
        X_dummy = np.empty((self.n_gtTracks, 1, self.nFrames, 3))
        X_dummy[:] = np.nan
        
        X = self.mat_ground[np.newaxis, :, :, :]
        Y = self.mat_tracks[:, np.newaxis, :, :]

        Ytilde = np.concatenate((Y,X_dummy), axis = 0).astype(np.float32)

        self.cost_XY = self.__costMatrixBetweenSets(X, Ytilde)
        
        self.YIndex, self.XIndex = linear_sum_assignment(self.cost_XY)
        
        self.X_opt = X[0, self.XIndex, :, :]
        self.Y_opt = Ytilde[self.YIndex, 0, :, :]
        
        
    def __calcAlphaAndBeta(self):
        d_XY = np.sum(self.cost_XY[self.YIndex, self.XIndex])    
        d_XD = self.__getDummyCosts(self.n_gtTracks)

        Ybar_tracks = (set(np.arange(self.n_dTracks + self.n_gtTracks)) - set(self.YIndex)) & set(np.arange(self.n_dTracks))
        
        if Ybar_tracks == set():
            d_YbarD = 0
            self.Ybar = 0
        else:
            d_YbarD = self.__getDummyCosts(len(Ybar_tracks))
            self.Ybar = self.mat_tracks[list(Ybar_tracks), :, :]
        
        self.metrics['alpha'] = (1 - (d_XY/d_XD))
        self.metrics['beta'] = (d_XD - d_XY)/(d_XD + d_YbarD)
        
        
    def __particleAssociationMetrics(self):
        mat = np.sqrt(np.sum((self.X_opt - self.Y_opt)**2, axis = 2))
        mat[np.isnan(mat)] = self.thresh
        
        #Number of position assignments with a cost less than thresh (i.e. correct assignments)
        self.metrics['par_tp'] = len(mat[mat < self.thresh].ravel())
        
        #Number of used dummy observations (tracks times number of frames) and 'missing' real observations
        self.metrics['par_fn'] = (np.sum(self.YIndex >= self.n_dTracks)*self.nFrames) + np.sum(np.isnan(self.Y_opt))/3
    
        #Number of position assignments with cost greater than threshold. And spurious observations from detected tracks. 
        self.metrics['par_fp'] = (len(mat[mat >= self.thresh]) - self.metrics['par_fn'])
        
        if not isinstance(self.Ybar, int):
            self.metrics['par_fp'] += np.sum(~np.isnan(self.Ybar))/3
           
        self.metrics['par_jaccard'] = self.metrics['par_tp']/(self.metrics['par_tp'] + self.metrics['par_fn'] + self.metrics['par_fp'])
        

    def __trackAssociationMetrics(self):
        
        #Number of selected tracks from Y
        self.metrics['track_tp'] = np.sum(self.YIndex < self.n_dTracks) 
        
        #Number of selected tracks which are dummy tracks
        self.metrics['track_fn'] = np.sum(self.YIndex > self.n_dTracks)
        
        #number of tracks from algorithm not used
        if not isinstance(self.Ybar, int):
            self.metrics['track_fp'] = self.Ybar.shape[0]
        else:
            self.metrics['track_fp'] = self.Ybar
        
        self.metrics['track_jaccard'] = self.metrics['track_tp']/(self.metrics['track_tp'] + self.metrics['track_fn'] + self.metrics['track_fp'])
        

    def __localisationMetrics(self):
        #1d array of all costs
        costs = np.sqrt(np.sum((self.X_opt - self.Y_opt)**2, axis = 2)).ravel()
    
        #Remove nan and values greater than thresh to leave only true positives
        costs = costs[~np.isnan(costs)]
        costs = costs[costs < self.thresh]
        
        self.RMSE = np.sqrt(np.sum(np.square(costs))/len(costs))
        self.minn = np.min(costs)
        self.maxx = np.max(costs)
        self.sd = np.std(costs)
        
        
    def __dfToArray(self, df):
        nTracks = len(pd.unique(df.loc[:, 'trackID']))
        mat = np.empty((nTracks, self.nFrames, 3))
        mat[:] = np.nan
        
        i = 0
        for track in pd.unique(df.loc[:, 'trackID']):
            dfTrack = df[df['trackID'] == track]
            frames = dfTrack['timePoint']
            mat[i, frames, :] = dfTrack.loc[:, ['x', 'y', 'z']]
            i += 1
            
        return mat.astype(np.float32)

    
    def __costMatrixBetweenSets(self, thetaX, thetaY):
        
        if thetaY.shape[0]*thetaX.shape[1]*thetaX.shape[2]*thetaX.shape[3]*4 > psutil.virtual_memory()[1]:
            print('mat too big doing calculation in stages')
            mat = np.zeros((thetaY.shape[0], thetaX.shape[1]), dtype = 'float32')
            
            for t in range(thetaX.shape[2]):
                matAtT = np.sqrt(np.sum((thetaX[:, :, t, :] - thetaY[:, :, t, :])**2, axis = 2))
                matAtT[np.isnan(matAtT)] = self.thresh
                matAtT[matAtT > self.thresh] = self.thresh
                
                mat += matAtT
            
            return mat
        
        mat = np.sqrt(np.sum((thetaX - thetaY)**2, axis = 3))
        mat[np.isnan(mat)] = self.thresh
        mat[mat > self.thresh] = self.thresh
        
        return np.sum(mat, axis = 2)
    
    
    def __getDummyCosts(self, numTracks):
        return self.nFrames*numTracks*self.thresh
        
            

class LoyaltyAnalysis:
    
    def __init__(self, isSub, dF_ground = None, dF_tracks = None, thresh = 5, mat_ground = None, mat_tracks = None):
        if not isSub:
            if (mat_ground is None) and (mat_tracks is None) and (dF_ground is None) and (dF_tracks is None):
               print('you must supply array or dataframe for ground truth and tracking')
            else:
                self.df_ground = dF_ground.copy()
                self.df_tracks = dF_tracks.copy()
                
                self.n_gtTracks = len(pd.unique(self.df_ground.loc[:, 'trackID']))
                self.n_dTracks = len(pd.unique(self.df_tracks.loc[:, 'trackID']))
                
                self.nFrames = len(pd.unique(self.df_ground.loc[:, 'timePoint']))
                self.thresh = thresh
                
                self.mat_ground = self.__dfToArray(self.df_ground)
                self.mat_tracks = self.__dfToArray(self.df_tracks)
                
                self.metrics = {}
        
        self.__loyaltyMetrics()
        
        
    def __loyaltyMetrics(self):    
        X = self.mat_ground[np.newaxis, :, :, :]
        Y = self.mat_tracks[:, np.newaxis, :, :]
        
        if Y.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]*4 > psutil.virtual_memory()[1]:
            print('mat too big doing calculation in stages')
            mat_euclidean = np.zeros((Y.shape[0], X.shape[1], X.shape[2]), dtype = 'float32')
            
            for t in range(X.shape[2]):
                mat_euclidean[:, :, t] = np.sqrt(np.sum((X[:, :, t, :] - Y[:, :, t, :])**2, axis = 2))

        else:
            mat_euclidean = np.sqrt(np.sum((X - Y)**2, axis = 3))
        
        #This is needed as nanargmin cannot deal with all nan rows. If row is all nan leave be otherwise populate with the lowest cost
        nanMask = np.tile(np.nanmin(mat_euclidean, axis=1, keepdims = True), (1, self.n_gtTracks, 1))
        
        #replace all nan rows with all inf rows. Then find the col index i.e. lowest cost GT to assign
        self.minVals = np.nanargmin(np.where(np.isnan(nanMask), np.inf, mat_euclidean), axis = 1)
        
        #set vales which are nan back to nan. This is used to determine times at which tracks don't exist.
        #min vals is 2d. Rows are track ID and cols are time. Value is lowest cost GT track
        self.minVals = np.where(np.isnan(nanMask[:, 0, :]), np.nan, self.minVals)
        
        #for each row count columns till value changes
        self.nConsecutiveAssignments = np.apply_along_axis(self.__framesToChange, 1, self.minVals)
        self.metrics['aveFramesOnFirstParticle'] = self.nConsecutiveAssignments.mean()
        self.metrics['propLoyalTracks'] = len(self.nConsecutiveAssignments[self.nConsecutiveAssignments == self.nFrames - 1])/len(self.nConsecutiveAssignments)
        
        #for each row count columns till value changes. And then to next change etc
        numFramesOnParticle = np.apply_along_axis(self.__framesOnParticle, 1, self.minVals)
        self.metrics['aveFramesOnParticle'] = np.nanmean(numFramesOnParticle)

        #check no duplicate assignments
        
        # s = np.sort(self.minVals, axis = 0)
        # duplicate_vals = s[:-1][s[1:] == s[:-1]]
        # print(duplicate_vals)
        
        #check correct number of assignments

        # for row in range(self.minVals.shape[0]):
        #     val, count = np.unique(self.minVals[row], return_counts=True)
        #     nFram = np.sum(count)
        #     nConsec = np.nansum(numFramesOnParticle[row, :])
        #     nswitch = len(numFramesOnParticle[row, :][~np.isnan(numFramesOnParticle[row, :])])    
            
        #     if nFram != nConsec + nswitch:
        #         #print(self.minVals[row][~np.isnan(self.minVals[row])])
        #         print('problem')
        #         print(self.minVals[row])
        #         print(numFramesOnParticle[row])
        #         self.__framesOnParticleDebug(self.minVals[row])
        #         print('\n\n')
                

        #drop nan and flatten so left with 1d array of number of consecutive frames algorithm tracked a GT particle
        numFramesOnParticle_flat = numFramesOnParticle.ravel()
        numFramesOnParticle_flat_nonan = numFramesOnParticle_flat[~np.isnan(numFramesOnParticle_flat)]
        self.numFramesOnParticle = numFramesOnParticle_flat_nonan
        
    def __framesToChange(self, arr_1d):
        arr_1d = arr_1d[~np.isnan(arr_1d)]
        return next((i for i, x in enumerate(arr_1d) if x != arr_1d[0]), len(arr_1d)) - 1
    
    
    def __framesOnParticle(self, arr_1d):
        arr_1d = arr_1d[~np.isnan(arr_1d)]
        start_frame = 0
        lengthOfRuns = np.empty((self.nFrames))
        lengthOfRuns[:] = np.nan
        i = 0
        originalSize = len(arr_1d)
        while start_frame < originalSize:
            
            numSteps = self.__framesToChange(arr_1d)
            lengthOfRuns[i] = numSteps 
            start_frame += numSteps + 1
            arr_1d = arr_1d[numSteps + 1:]
            i += 1
        return lengthOfRuns    
                        
    def __dfToArray(self, df):
        nTracks = len(pd.unique(df.loc[:, 'trackID']))
        mat = np.empty((nTracks, self.nFrames, 3))
        mat[:] = np.nan
        
        i = 0
        for track in pd.unique(df.loc[:, 'trackID']):
            dfTrack = df[df['trackID'] == track]
            frames = dfTrack['timePoint']
            mat[i, frames, :] = dfTrack.loc[:, ['x', 'y', 'z']]
            i += 1
            
        return mat
    
    def plotLoyalty(self):
        current_cmap = matplotlib.cm.get_cmap('hot')
        current_cmap.set_bad(color='blue')
    
        plt.figure()
        plt.imshow(self.minVals, cmap='hot', interpolation='nearest', aspect=self.minVals.shape[1]/self.minVals.shape[0])
        plt.title('Loyalty plot: lowest cost associations against time')
        plt.xlabel('time (frames)')
        plt.ylabel('detected track ID')
        cbar = plt.colorbar()
        cbar.set_label('ID of lowest cost ground truth track')
        
    def plotHistConsecutiveFrames(self, title, nbins = 20):
        
        #1 is added so range spans 1:1000 in stead of 0:999
        
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle('Histogram of loyalty duration: ' + title)
        hist, bins, _ = axs[0].hist(self.numFramesOnParticle + 1, bins = nbins)
        axs[0].set_xlim((0, 1000))
        axs[0].set_xlabel('number of consecutive frames')
        axs[0].set_ylabel('frequency')
        
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        
        axs[1].hist(self.numFramesOnParticle + 1, bins=logbins)
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlim((1, 1001))
        axs[1].set_title('log-log')
        axs[1].set_xlabel('number of consecutive frames')
        axs[1].set_ylabel('frequency')

class TrackAnalysis(CorrespondanceAnalysis, LoyaltyAnalysis):
    def __init__(self, dF_tracks, dF_ground = None, pixsToMicrons = 954.21/100, thresh = 5):
        self.timeAveMSDs = None
        self.ensembleMSD = None
        self.timeLags = None
        
        self.timeAveMSDs_ground = None
        self.ensembleMSD_ground = None
        self.timeLags_ground = None
        
        if np.any(dF_ground == None):
            self.df_tracks = dF_tracks.copy()

            self.n_dTracks = len(pd.unique(self.df_tracks.loc[:, 'trackID']))
            
            self.nFrames = len(pd.unique(self.df_tracks.loc[:, 'timePoint']))
            self.thresh = thresh

            self.mat_tracks = self.__dfToArray(self.df_tracks)
            
            self.metrics = {}
        else:
            super().__init__(dF_ground = dF_ground, dF_tracks = dF_tracks, thresh = thresh, isSub = True)
        
        temp = sorted(pd.unique(self.df_tracks['t']))
        self.dt = temp[1] - temp[0]
        self.pixsToMicrons = pixsToMicrons
    
    def __dfToArray(self, df):
        nTracks = len(pd.unique(df.loc[:, 'trackID']))
        mat = np.empty((nTracks, self.nFrames, 3))
        mat[:] = np.nan
        
        i = 0
        for track in pd.unique(df.loc[:, 'trackID']):
            dfTrack = df[df['trackID'] == track]
            frames = dfTrack['timePoint']
            mat[i, frames, :] = dfTrack.loc[:, ['x', 'y', 'z']]
            i += 1
            
        return mat
    
    def timeAveMSD(self, tracK, timeLags = None):
        
        
        track = tracK[~np.isnan(tracK).any(axis = 1)]        
        if np.any(timeLags == None):
            maxTimeLag = track.shape[0]/2
            timeLags = np.arange(1, maxTimeLag, maxTimeLag/20).astype('int')

        MSDs = [np.sum(np.square(np.sum(np.absolute(track[timeLag:, :] - track[:-timeLag, :]), axis = 1)), axis = 0)/len(track[timeLag:, 0]) for timeLag in timeLags if timeLag > 0]
        if MSDs[-1] == np.nan:
            print('MSD no calculated correctly')
        
        if len(MSDs) < len(timeLags):
            MSDs = np.pad(MSDs, (0, len(timeLags) - len(MSDs)), 'constant')
        
        return MSDs
        
    def ensembleAveMSD(self, matOfTracks, nTimeLags = 20):
        
        timeLags = np.zeros((matOfTracks.shape[0], nTimeLags), dtype=int)
        timeAveMSDs = np.zeros((matOfTracks.shape[0], nTimeLags), dtype=float)
        
        for row in range(matOfTracks.shape[0]):
            
            maxTimeLag = np.floor(np.count_nonzero(~np.isnan(matOfTracks[row, :, 0]))/2).astype('int')
            if maxTimeLag < nTimeLags:
                timeLags[row, :maxTimeLag] = np.arange(maxTimeLag) + 1
            else:
                timeLags[row] = np.arange(1, maxTimeLag, maxTimeLag/nTimeLags).astype('int')
            timeAveMSDs[row] = self.timeAveMSD(matOfTracks[row], timeLags[row])
            
        ensembleMSD = np.mean(timeAveMSDs, axis = 0)
        
        return timeLags, timeAveMSDs, ensembleMSD
        
        
    def plotEnsembleAve(self, ground = False):
        
        if ground == False:
            print('plotting tracks')
            if np.any(self.ensembleMSD == None):
                self.timeLags, self.timeAveMSDs, self.ensembleMSD = self.ensembleAveMSD(self.mat_tracks)
            self.__pltMSD(self.timeLags[0], self.ensembleMSD)
                
        elif ground == True:
            print('plotting ground')
            if np.any(self.ensembleMSD_ground == None):
                self.timeLags_ground, self.timeAveMSDs_ground, self.ensembleMSD_ground = self.ensembleAveMSD(self.mat_ground)
            self.__pltMSD(self.timeLags_ground[0], self.ensembleMSD_ground)
                
    def plotAllTimeAve(self, ground = False):
        
        if ground == False:
            print('plotting tracks')
            if np.any(self.ensembleMSD == None):
                self.timeLags, self.timeAveMSDs, self.ensembleMSD = self.ensembleAveMSD(self.mat_tracks)
            self.__pltManyMSD(self.timeLags, self.timeAveMSDs)
                
        elif ground == True:
            print('plotting ground')
            if np.any(self.ensembleMSD_ground == None):
                self.timeLags_ground, self.timeAveMSDs_ground, self.ensembleMSD_ground = self.ensembleAveMSD(self.mat_ground)
            self.__pltManyMSD(self.timeLags_ground, self.timeAveMSDs_ground)
        
        
    def __pltMSD(self, timeLags, ensembleMSD):
        
        plt.figure()
        plt.title('ensemble MSD')
        plt.loglog(timeLags, ensembleMSD)
        
    def __pltManyMSD(self, timeLags, timeAveMSDs, title = '', timeU = 's', distU = 'um'):

        
        row, col = np.where(timeLags == 0)
        timeLags = timeLags.astype('float')
        timeLags[row, col] = np.nan
        timeAveMSDs[row, col] = np.nan
        
        plt.figure()
        
            
        if timeU == 's':
            #concert frames to seconds
            print('converting frames to seconds')
            timeLags[:, :] = timeLags[:, :] * self.dt
        elif timeU == 'frames':
            #concert frames to seconds
            timeU = r'$frames$'
            
        if distU == 'um':
            #convert pi^2 to um^2
            distU = r'$\mu m$'
            
            timeAveMSDs[:, :] = timeAveMSDs[:, :] * pow(self.pixsToMicrons, 2)
        elif distU == 'pi':
            #concert frames to seconds
            distU = r'$pi$'
            
        
        plt.title('time ave MSD')
        plt.loglog(timeLags.T, timeAveMSDs.T)
        plt.xlabel(r'$\tau ($' + timeU + r'$)$')
        plt.ylabel(r'$msd(\tau)$' + ' ' +  r'$($' + distU + r'$^2)$')
        plt.grid(which = 'major')
        plt.grid(which = 'minor', linestyle = '--')
        
            
    def plotHistTrackLengths(self, df, title, save = False, fileDir = ""):
        
        dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
        dfTrackIDs.columns = ['Histogram of track lengths']
        
        fig, ax = plt.subplots()
        dfTrackIDs.hist(ax=ax, bins=100, bottom=0.1)
        plt.title(title)
        ax.set_yscale('log')
        ax.set_xscale('log')
        if save:
            plt.savefig(fileDir)
            
    def plot_tracks(self, df, trackIDs):
    
        for trackID in trackIDs:
            track = df.loc[df['trackID'] == trackID, ['x', 'y', 'timePoint']]
            
            sns.relplot(data = track, x = 'x', y = 'y',sort = False ,kind = 'line', estimator=None, lw = 0.5)
            plt.show()
            sns.relplot(data = track, x = 'x', y = 'y', kind = 'scatter', hue = 'timePoint' , estimator=None, lw = 0.5)
            plt.show()
            
    def plotAllTracks(self, dF, title, save = False, fileDir = ""):
        df = dF.copy()
        
        for trackID in pd.unique(df['trackID']):
            track = df.loc[df['trackID'] == trackID, :]
            df.loc[df['trackID'] == trackID, ['x', 'y']] = track.loc[:,['x', 'y']] - track.loc[track.index[0], ['x', 'y']]
        
        
        sns.relplot(data = df, x = 'x', y = 'y',sort = False ,kind = 'line', hue = 'trackID', estimator=None, lw = 0.5, legend = False)
        plt.title(title)
        #plt.axis((-150,150,-150,150))
        plt.xlabel('x - x0')
        plt.ylabel('y - y0')
        plt.show()
        if save:
            plt.savefig(fileDir)
    
        
        
    
    

    
