# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:37:48 2020

@author: j72687wm
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from kalmanTracker import KalmanFilter

inff = 1000000000

class IMM:
    
    __counter = 0
    
    @staticmethod
    def filterInstances():
        return KalmanFilter.__counter 
    
    def __init__(self, KFs, M, mu):
        
        self.X_comb = np.nan
        self.P_comb = np.nan
        
        self.M = M
        self.filters = KFs
        self.probas_predict = mu
        self.probas_update = [None] * len(KFs)
        self.n = len(KFs)
        
    def predict(self):
        for kf in self.filters:
            kf.predict()
    
    def update(self, Y):
        for kf in self.filters:
            kf.update(Y)
            
    def predictProbabilities(self):
        for i in range(self.n):
            self.probas_predict[i] = sum([self.M[i, j]*self.probas_update[j] for j in range(self.n)]) 
        
        
    def updateProbabilities(self):
        for i in range(self.n):
            self.probas_update[i] = (self.probas_predict[i]*self.filters[i].likelihood)/sum([self.probas_predict[j]*self.filters[j].likelihood for j in range(self.n)])
            
        #print(self.probas_update)
    
    def mixEstimates(self):
        #output
       
        self.X_comb = np.sum(np.stack([self.filters[i].X*self.probas_update[i] for i in range(self.n)]), axis = 0)
        self.P_comb = np.sum(np.stack([self.probas_update[i]*(self.filters[i].P + (self.X_comb - self.filters[i].X).dot((self.X_comb - self.filters[i].X).T)) for i in range(self.n)]), axis = 0)
                                                  
        self.predictProbabilities()
        X_update = [self.filters[j].X for j in range(self.n)]
        for i in range(self.n):
            
            mixingProbs = np.array((self.M[i, :]*[self.probas_update[j] for j in range(self.n)])/(self.probas_predict[i]))
            mixingProbs = mixingProbs[:, np.newaxis, np.newaxis]

            self.filters[i].X = np.sum(np.stack([mixingProbs[j]*self.filters[j].X for j in range(self.n)]), axis = 0)
            self.filters[i].P = np.sum(np.stack(mixingProbs*[self.filters[j].P  + (X_update[j] - self.filters[i].X).dot((X_update[j] - self.filters[i].X).T) for j in range(self.n)]), axis = 0)
			
def calcCostMatrixIMM(window, tracks, pointRows, immDict, x):

    costMatrix = np.zeros((len(tracks), len(pointRows)))

    trackVectorRW = np.zeros((len(tracks), 1, len(x)))
    trackVectorCV = trackVectorRW.copy()
    trackVectorCA = trackVectorRW.copy()
    trackVectors = [trackVectorRW, trackVectorCV, trackVectorCA]
    
    i = 0
    for track in tracks:
        j = 0
        for kf in immDict[track].filters:
            trackVectors[j][i, 0, :] = np.dot(kf.H, kf.X).T
            j += 1
        i += 1
    
    pointVector = np.zeros((1, len(pointRows), len(x)))
    pointVector[0, :, :] = window.loc[pointRows,x]

    costMatrices = [np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2)) for trackVector in trackVectors]
    
    costMatrix = np.amin(np.stack(costMatrices), axis = 0)
    return costMatrix
	
def twoFrameLinking_withIMM(dF, lb, maxDiff, Fs, Ps, Qs, R, H, M, mu, n_frames, modes, x):
    df = dF.copy()             
    
    binnedCosts = []
    halfway = False
    
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['frame'] == 0, 'trackID'] = df.loc[df['frame'] == 0, 'pointID'] + 1
            IMM_Dict = {}
            
            for track in pd.unique(df.loc[df['frame'] == 0, 'trackID']):
                y = df.loc[((df['frame'] == 0) & (df['trackID'] == track)), x].values.ravel()
                X_o = y.dot(H).T
				
				#make a new calman filter object and store within a dictionary using track ID as key
                KF_for_IMM = []
                for i in range(len(Fs)):
	                KF_for_IMM.append(KalmanFilter(X_o, Fs[i], Qs[i], R, Ps[i], H, modes[i] + str(track)))
                IMM_Dict[track] = IMM(KF_for_IMM, M, mu)
                              
        
            print("Assigned tracks")
            continue
        
        endFrame = currentFrame - 1
        startFrame = currentFrame - 1 - lb
        
        if startFrame < 0:
            startFrame = 0
        
        window = df.loc[(df['frame'] >= startFrame) & (df['frame'] <= endFrame),:]
        mostRecentTracks = []
        
        for track in pd.unique(window.trackID):
            if np.isnan(track):
                continue      
            #Gets the index of the row which matches the trackID and has the largest tp.
            #If multiple matches at one timepoint only returns the first
            index = window.loc[window['trackID'] == track,'frame'].idxmax()
            mostRecentTracks.append((track, index))
            
        if len(mostRecentTracks) < 1:
            print('no active tracks')
            pointIDs = df.loc[df['frame'] == currentFrame].index
            if len(pointIDs) < 1:
                print('and no detections')
                continue
            else:
                nTracks = np.amax(df['trackID'])
                newTracks = np.arange(nTracks + 1, nTracks + len(pointIDs) + 1)
                df.loc[pointIDs, 'trackID'] = newTracks
                
                for newTrack in newTracks:
                    y = df.loc[df['trackID'] == newTrack, x].values.ravel()
                    X_o = y.dot(H).T
                             
                    #make a new kalman filter object and store within a dictionary using track ID as key
                    KF_for_IMM = []
                    for i in range(len(Fs)):
                        KF_for_IMM.append(KalmanFilter(X_o, Fs[i], Qs[i], R, Ps[i], H, modes[i] + str(track)))
                    IMM_Dict[track] = IMM(KF_for_IMM, M, mu)
                continue
        
        #print(mostRecentTracks)
        activeTracks, trackIDs = zip(*mostRecentTracks)
        
          #Make prediction for all active kalman filters
        for activeTrack in activeTracks:
            IMM_Dict[activeTrack].predict()
            
            
        trackIDs = np.asarray(trackIDs)
        activeTracks = np.asarray(activeTracks)
        
        #update window to include current frame
        window = df.loc[(df['frame'] >= startFrame) & (df['frame'] <= currentFrame),:]
        pointIDs = df.loc[df['frame'] == currentFrame].index
        
        costMatrix = calcCostMatrixIMM(window, activeTracks, pointIDs, IMM_Dict, x)  
        
        costMatrix[costMatrix > maxDiff] = inff
        #assignments in the form row index, col index
        
        try:
            trackIndex, pointIndex = linear_sum_assignment(costMatrix)
        except ValueError:
            print(costMatrix)
            break
        
        #rows and cols not to be assigned. I've converted to sets to do this.
        rows, cols = np.nonzero(costMatrix > maxDiff)
        aboveThresh = set(zip(rows, cols))
        assignments = set(zip(trackIndex, pointIndex))
        
        #remove assignments above threshold
        assignedAndBelowThresh = assignments - aboveThresh
        assignedAndBelowThresh = list(zip(*assignedAndBelowThresh))
        
        if len(assignedAndBelowThresh) == 0:
            print('No observations within gates')
            unassignedKPs = list(set(pointIDs))   
            
        else:
            points = list(assignedAndBelowThresh[1])
            tracks = list(assignedAndBelowThresh[0])
            unassignedKPs = list(set(pointIDs) - set(pointIDs[points]))
        
            df.loc[pointIDs[points], 'trackID'] = activeTracks[tracks]
            
            
        nTracks = np.amax(df['trackID'])
        newTracks = np.arange(nTracks + 1, nTracks + len(unassignedKPs) + 1)
        df.loc[unassignedKPs, 'trackID'] = newTracks
        
        #initialise new IMM filters
        for newTrack in newTracks:
            y = df.loc[df['trackID'] == newTrack, x].values.ravel()
            X_o = y.dot(H).T
                              
            #make a new kalman filter object and store within a dictionary using track ID as key
            KF_for_IMM = []
            for i in range(len(Fs)):
                KF_for_IMM.append(KalmanFilter(X_o, Fs[i], Qs[i], R, Ps[i], H, modes[i] + str(track)))
            IMM_Dict[newTrack] = IMM(KF_for_IMM, M, mu)
        
        #update existing kalman filters with assigned readings
        if len(assignedAndBelowThresh) > 0:
            for track, point in zip(activeTracks[tracks], pointIDs[points]):
                y = df.loc[point, x].values.ravel()
                y = np.array([[y[0], y[1], y[2], y[3]]]).T
                IMM_Dict[track].update(y)
                IMM_Dict[track].updateProbabilities()
                IMM_Dict[track].mixEstimates()
        
		#remove tracks with fewer than 20 occurances    
		#dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
		#tracksToRemove = dfTrackIDs.loc[dfTrackIDs['pointID'] < 20, :]
		#df_excessTracksRemoved = df.copy()
		#df_excessTracksRemoved.loc[df_excessTracksRemoved['trackID'].isin(tracksToRemove.index), 'trackID'] = np.nan
    return df
	
def main(df_kp, n_frames, lb, maxDiff, Fs, Ps, Qs, R, H, M, mu, modes, x):
    
    if (df_kp is None or n_frames is None):
        return None
    
    df_tracks_IMM = twoFrameLinking_withIMM(df_kp, lb, maxDiff, Fs, Ps, Qs, R, H, M, mu, n_frames, modes, x)
    
    
    return df_tracks_IMM
    


if __name__ == '__main__':
    df_kp = None
    im = None
    main()