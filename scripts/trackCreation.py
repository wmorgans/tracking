# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:07:40 2020

@author: j72687wm
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag, expm, det, inv
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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
    
    
class KalmanFilter:
    
    __counter = 0
    
    
    def __init__(self, X_o, F, Q, R, P, H, name):
        self.name = name
        self.X = X_o
        self.P = P
        self.F = F
        
        #self.w_k = np.array()
        self.Q = Q
        #self.z_k = np.array()
        self.R = R
        
        #H is not always square !!!!! Needs to map X to Y (rows equal to length of Y and cols 
        #equal to length of X)
        self.H = H
        
        self.likelihood = np.nan
        self.K = np.nan
        
        type(self).__counter += 1
        
    @staticmethod
    def filterInstances():
        return KalmanFilter.__counter   
    
    def predict(self):
        #print('in predict')

        self.X = np.dot(self.F, self.X)
        self.P = (self.F.dot(self.P).dot(np.transpose(self.F))) + self.Q
        
    def update(self, Y):
        #resiadual or innovation and variance of innovation
        #v = Y - HX
        #S = R + HPH^(t))
        
        v = Y - (self.H.dot(self.X))
        S = self.R + (self.H.dot(self.P).dot(np.transpose(self.H)))
        S_inv = inv(S)
        #K = P.H.(H.P.H^(t)+R)^(-1)
        self.K = self.P.dot(self.H.T).dot(S_inv)
        #X = X + K.v
        self.X = self.X + self.K.dot(v)
        
        #self.P = self.P + self.K.dot(S).dot(np.transpose(self.K))
        I = np.identity(self.P.shape[0])
        self.P = (I - self.K.dot(self.H)).dot(self.P).dot((I - self.K.dot(self.H)).T) + self.K.dot(self.R).dot(self.K.T)
        self.P = (self.P + self.P.T)/2
        #likelihood
        #det(2xpixS))^(-0.5)xexp[(-0.5)v^T.S^-1.v]
        self.likelihood = np.power(det((2*np.pi)*S), -1/2)*expm((-0.5)*(v.T).dot(S_inv).dot(v)).item()
        

def calcCostMatrix(window, trackRows, pointRows):

    costMatrix = np.zeros((len(trackRows), len(pointRows)))
    trackVector = np.zeros((len(trackRows), 1, 3))
    trackVector[:, 0, :] = window.loc[trackRows,['x', 'y', 'a']]
    
    
    pointVector = np.zeros((1, len(pointRows), 3))
    pointVector[0, :, :] = window.loc[pointRows,['x', 'y', 'a']]
    
    
    costMatrix = np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2))
    
    return costMatrix

def calcCostMatrixKalman(window, tracks, pointRows, kalmanDict):

    costMatrix = np.zeros((len(tracks), len(pointRows)))
    
    trackVector = np.zeros((len(tracks), 1, 3))
    
    i = 0
    for track in tracks:
        trackVector[i, 0, :] = np.dot(kalmanDict[track].H, kalmanDict[track].X).T
        i += 1
    
    pointVector = np.zeros((1, len(pointRows), 3))
    pointVector[0, :, :] = window.loc[pointRows,['x', 'y', 'a']]
    
    costMatrix = np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2))
    
    return costMatrix

def calcCostMatrixIMM(window, tracks, pointRows, immDict):

    costMatrix = np.zeros((len(tracks), len(pointRows)))

    trackVectorRW = np.zeros((len(tracks), 1, 3))
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
    
    pointVector = np.zeros((1, len(pointRows), 3))
    pointVector[0, :, :] = window.loc[pointRows,['x', 'y', 'a']]

    costMatrices = [np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2)) for trackVector in trackVectors]
    
    costMatrix = np.amin(np.stack(costMatrices), axis = 0)
    return costMatrix


def twoFrameLinking(dF, lb, maxDiff, n_frames):
    df = dF.copy()
    binnedCosts = []
    halfway = False
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['timePoint'] == 0, 'trackID'] = df.loc[df['timePoint'] == 0, 'pointID'] + 1
            print("Assigned tracks")
            continue
        
        endFrame = currentFrame - 1
        startFrame = currentFrame - 1 - lb
        
        if startFrame < 0:
            startFrame = 0
        
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= endFrame),:]
        mostRecentTracks = []
        
        for track in pd.unique(window.trackID):
            if np.isnan(track):
                continue      
            #Gets the index of the row which matches the trackID and has the largest tp.
            #If multiple matches at one timepoint only returns the first
            index = window.loc[window['trackID'] == track,'timePoint'].idxmax()
            mostRecentTracks.append((track, index))
        
        #print(mostRecentTracks)
        if len(mostRecentTracks) < 1:
            print('no active tracks')
            pointIDs = df.loc[df['timePoint'] == currentFrame].index
            if len(pointIDs) < 1:
                print('and no detections')
                continue
            else:
                nTracks = np.amax(df['trackID'])
                newTracks = np.arange(nTracks + 1, nTracks + len(pointIDs) + 1)
                df.loc[pointIDs, 'trackID'] = newTracks
                continue
            
        activeTracks, trackIDs = zip(*mostRecentTracks)
        trackIDs = np.asarray(trackIDs)
        activeTracks = np.asarray(activeTracks)
        
        #update window to include current frame
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= currentFrame),:]
        pointIDs = df.loc[df['timePoint'] == currentFrame].index
        
        if len(pointIDs) < 1:
            print('no observations at current time step')
            continue
        
        costMatrix = calcCostMatrix(window,trackIDs, pointIDs)       
        costMatrix[costMatrix > maxDiff] = inff
        #assignments in the form row index, col index
        trackIndex, pointIndex = linear_sum_assignment(costMatrix)
        
        #rows and cols not to be assigned
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
            #print('number of assignments:', str(len(pointIDs[points])))
            
            
        nTracks = np.amax(df['trackID'])
        newTracks = np.arange(nTracks + 1, nTracks + len(unassignedKPs) + 1)
        df.loc[unassignedKPs, 'trackID'] = newTracks
        
        #print('number of new tracks:', str(len(unassignedKPs)))
        
    #remove tracks with fewer than 20 occurances    
    #dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
    #tracksToRemove = dfTrackIDs.loc[dfTrackIDs['pointID'] < 20, :]
    #df_excessTracksRemoved = df.copy()
    #df_excessTracksRemoved.loc[df_excessTracksRemoved['trackID'].isin(tracksToRemove.index), 'trackID'] = np.nan
        
        cm = np.ravel(costMatrix[trackIndex, pointIndex]) 
        #make sure those above threshold added to last bin
        cm[cm  == inff] = maxDiff + 1          
        histVals, histBins = np.histogram(cm, bins = 16, range = (0, 16*maxDiff/15))
        histVals = histVals/np.sum(histVals)
        binnedCosts.append(histVals)
        
    return df, binnedCosts, histBins



def twoFrameLinking_withKalman(dF, lb, maxDiff, F, P, Q, R, H, name, n_frames):
    df = dF.copy()
    binnedCosts = []
    halfway = False
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
            
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['timePoint'] == 0, 'trackID'] = df.loc[df['timePoint'] == 0, 'pointID'] + 1
            kalmanDict = {}
            
            for track in pd.unique(df.loc[df['timePoint'] == 0, 'trackID']):
                y = df.loc[((df['timePoint'] == 0) & (df['trackID'] == track)), ['x', 'y', 'a']].values.ravel()
                X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                              
                #make a new calman filter object and store within a dictionary using track ID as key
                kalmanDict[track] = KalmanFilter(X_o, F, Q, R, P, H, name + "_" + str(track))
            print("Assigned tracks")
            continue
        
        endFrame = currentFrame - 1
        startFrame = currentFrame - 1 - lb
        
        if startFrame < 0:
            startFrame = 0
        
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= endFrame),:]
        mostRecentTracks = []
        
        for track in pd.unique(window.trackID):
            if np.isnan(track):
                continue      
            #Gets the index of the row which matches the trackID and has the largest tp.
            #If multiple matches at one timepoint only returns the first
            index = window.loc[window['trackID'] == track,'timePoint'].idxmax()
            mostRecentTracks.append((track, index))
            
        if len(mostRecentTracks) < 1:
            print('no active tracks')
            pointIDs = df.loc[df['timePoint'] == currentFrame].index
            if len(pointIDs) < 1:
                print('and no detections')
                continue
            else:
                nTracks = np.amax(df['trackID'])
                newTracks = np.arange(nTracks + 1, nTracks + len(pointIDs) + 1)
                df.loc[pointIDs, 'trackID'] = newTracks
                
                for track in newTracks:
                    y = df.loc[df['trackID'] == track, ['x', 'y', 'a']].values.ravel()
                    X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                                      
                    #make a new kalman filter object and store within a dictionary using track ID as key
                    kalmanDict[track] = KalmanFilter(X_o, F, Q, R, P, H, name + "_" + str(track))
                continue
        
        activeTracks, trackIDs = zip(*mostRecentTracks)
         #Make prediction for all active kalman filters
        for activeTrack in activeTracks:
            kalmanDict[activeTrack].predict()
            
            
        trackIDs = np.asarray(trackIDs)
        activeTracks = np.asarray(activeTracks)
        
        #update window to include current frame
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= currentFrame),:]
        pointIDs = df.loc[df['timePoint'] == currentFrame].index
        
        if len(pointIDs) < 1:
            print('no observations at current time step')
            continue
        
        costMatrix = calcCostMatrixKalman(window,activeTracks, pointIDs, kalmanDict)       
        costMatrix[costMatrix > maxDiff] = inff
        #assignments in the form row index, col index
        trackIndex, pointIndex = linear_sum_assignment(costMatrix)
        
        
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
        
        #initialise new kalman filters
        for track in newTracks:
            y = df.loc[df['trackID'] == track, ['x', 'y', 'a']].values.ravel()
            X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                              
            #make a new kalman filter object and store within a dictionary using track ID as key
            kalmanDict[track] = KalmanFilter(X_o, F, Q, R, P, H, name + "_" + str(track))
        
        if len(assignedAndBelowThresh) > 0:
            #update existing kalman filters with assigned readings
            for track, point in zip(activeTracks[tracks], pointIDs[points]):
                y = df.loc[point, ['x', 'y', 'a']].values.ravel()
                y = np.array([[y[0], y[1], y[2]]]).T
                kalmanDict[track].update(y)
                
        cm = np.ravel(costMatrix[trackIndex, pointIndex]) 
        #make sure those above threshold added to last bin
        cm[cm  == inff] = 81          
        histVals, histBins = np.histogram(cm, bins = 16, range = (0, 16*maxDiff/15))
        histVals = histVals/np.sum(histVals)
        binnedCosts.append(histVals)
        
    #remove tracks with fewer than 20 occurances    
    #dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
    #tracksToRemove = dfTrackIDs.loc[dfTrackIDs['pointID'] < 20, :]
    #df_excessTracksRemoved = df.copy()
    #df_excessTracksRemoved.loc[df_excessTracksRemoved['trackID'].isin(tracksToRemove.index), 'trackID'] = np.nan
    return df, binnedCosts, histBins

def twoFrameLinking_withIMM(dF, lb, maxDiff, F_R, P_R, Q_R,
                            F_V, P_V, Q_V, F_A, P_A, Q_A, R, H, M, mu, n_frames):
    df = dF.copy()             
    
    binnedCosts = []
    halfway = False
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['timePoint'] == 0, 'trackID'] = df.loc[df['timePoint'] == 0, 'pointID'] + 1
            IMM_Dict = {}
            
            for track in pd.unique(df.loc[df['timePoint'] == 0, 'trackID']):
                y = df.loc[((df['timePoint'] == 0) & (df['trackID'] == track)), ['x', 'y', 'a']].values.ravel()
                X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                              
                #make a new calman filter object and store within a dictionary using track ID as key
                RW = KalmanFilter(X_o, F_R, Q_R, R, P_R, H, "RW_" + str(track))
                CV = KalmanFilter(X_o, F_V, Q_V, R, P_V, H, "CV_" + str(track))
                CA = KalmanFilter(X_o, F_A, Q_A, R, P_A, H, "CA_" + str(track))
                IMM_Dict[track] = IMM([RW, CV, CA], M, mu)
            print("Assigned tracks")
            continue
        
        endFrame = currentFrame - 1
        startFrame = currentFrame - 1 - lb
        
        if startFrame < 0:
            startFrame = 0
        
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= endFrame),:]
        mostRecentTracks = []
        
        for track in pd.unique(window.trackID):
            if np.isnan(track):
                continue      
            #Gets the index of the row which matches the trackID and has the largest tp.
            #If multiple matches at one timepoint only returns the first
            index = window.loc[window['trackID'] == track,'timePoint'].idxmax()
            mostRecentTracks.append((track, index))
            
        if len(mostRecentTracks) < 1:
            print('no active tracks')
            pointIDs = df.loc[df['timePoint'] == currentFrame].index
            if len(pointIDs) < 1:
                print('and no detections')
                continue
            else:
                nTracks = np.amax(df['trackID'])
                newTracks = np.arange(nTracks + 1, nTracks + len(pointIDs) + 1)
                df.loc[pointIDs, 'trackID'] = newTracks
                
                for newTrack in newTracks:
                    y = df.loc[df['trackID'] == newTrack, ['x', 'y', 'a']].values.ravel()
                    X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                                      
                    #make a new kalman filter object and store within a dictionary using track ID as key
                    RW = KalmanFilter(X_o, F_R, Q_R, R, P_R, H, "RW_" + str(track))
                    CV = KalmanFilter(X_o, F_V, Q_V, R, P_V, H, "CV_" + str(track))
                    CA = KalmanFilter(X_o, F_A, Q_A, R, P_A, H, "CA_" + str(track))
                    IMM_Dict[newTrack] = IMM([RW, CV, CA], M, mu)
                continue
        
        #print(mostRecentTracks)
        activeTracks, trackIDs = zip(*mostRecentTracks)
        
          #Make prediction for all active kalman filters
        for activeTrack in activeTracks:
            IMM_Dict[activeTrack].predict()
            
            
        trackIDs = np.asarray(trackIDs)
        activeTracks = np.asarray(activeTracks)
        
        #update window to include current frame
        window = df.loc[(df['timePoint'] >= startFrame) & (df['timePoint'] <= currentFrame),:]
        pointIDs = df.loc[df['timePoint'] == currentFrame].index
        
        costMatrix = calcCostMatrixIMM(window, activeTracks, pointIDs, IMM_Dict)  
        
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
            y = df.loc[df['trackID'] == newTrack, ['x', 'y', 'a']].values.ravel()
            X_o = np.array([[y[0], 0, 0, y[1], 0, 0, y[2]]]).T
                              
            #make a new kalman filter object and store within a dictionary using track ID as key
            RW = KalmanFilter(X_o, F_R, Q_R, R, P_R, H, "RW_" + str(track))
            CV = KalmanFilter(X_o, F_V, Q_V, R, P_V, H, "CV_" + str(track))
            CA = KalmanFilter(X_o, F_A, Q_A, R, P_A, H, "CA_" + str(track))
            IMM_Dict[newTrack] = IMM([RW, CV, CA], M, mu)
        
        #update existing kalman filters with assigned readings
        if len(assignedAndBelowThresh) > 0:
            for track, point in zip(activeTracks[tracks], pointIDs[points]):
                y = df.loc[point, ['x', 'y', 'a']].values.ravel()
                y = np.array([[y[0], y[1], y[2]]]).T
                IMM_Dict[track].update(y)
                IMM_Dict[track].updateProbabilities()
                IMM_Dict[track].mixEstimates()
        
    #remove tracks with fewer than 20 occurances    
    #dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
    #tracksToRemove = dfTrackIDs.loc[dfTrackIDs['pointID'] < 20, :]
    #df_excessTracksRemoved = df.copy()
    #df_excessTracksRemoved.loc[df_excessTracksRemoved['trackID'].isin(tracksToRemove.index), 'trackID'] = np.nan
        cm = np.ravel(costMatrix[trackIndex, pointIndex]) 
        #make sure those above threshold added to last bin
        cm[cm  == inff] = 81          
        histVals, histBins = np.histogram(cm, bins = 16, range = (0, 16*maxDiff/15))
        histVals = histVals/np.sum(histVals)
        binnedCosts.append(histVals)
        if False:
            allActiveFilters = assignedAndBelowThresh + newTracks
            vel = []
            for filt in allActiveFilters:
                velX = filt.KF[2].X[1]
                velY = filt.KF[2].X[4]
                vel.append(np.sqrt(velX**2 + velY**2))
            vel = np.array(vel)
            histV, vBins = np.histogram(vel, bins = 15)
    
    return df, binnedCosts, histBins

def main(df_kp, n_frames):
    
    if (df_kp is None or n_frames is None):
        return
    #params
    lb = 1
    maxDiff = 25
    
    start_naive = timer()
    df_tracks, binnedCostsN, histBinsN = twoFrameLinking(df_kp, lb, maxDiff, n_frames)
    
    end_naive = timer()
    
    dt = 0.03
    #for measurement noise
    xvar = 1.5
    yvar = 1.5
    avar = xvar*yvar
    
    #for process noise
    sigP = 10.
    sigA = sigV = 40.
    M = np.array([[0.7, 0.15, 0.15],
                  [0.2, 0.7, 0.1],
                  [0.25, 0.25, 0.5]])
    mu = [0.8, 0.1, 0.1]
    
    #For RW
    F_R = np.array([[1., 0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  1, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 1]])
    
    P_R = np.array([[xvar, 0., 0,  0, 0, 0, 0],
                    [0,  0, 0,     0, 0, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  yvar, 0, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  0, 0, 0, avar]])
    
    Q_R = np.array([[sigP, 0., 0,  0, 0, 0, 0],
                    [0,  0, 0, 0, 0, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  sigP, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 1.]])
    
    #For const V
    F_V = np.array([[1., dt, 0,  0, 0, 0, 0],
                    [0,  1, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  1, dt, 0, 0],
                    [0,  0, 0,  0, 1, 0, 0],
                    [0,  0, 0,  0, 0, 0, 0],
                    [0,  0, 0,  0, 0, 0, 1]])
    
    P_V = np.array([[xvar, 0., 0,  0, 0, 0, 0],
                    [0,  100, 0,    0, 0, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  yvar, 0, 0, 0],
                    [0,  0, 0,  0,   100, 0, 0],
                    [0,  0, 0,  0,    0, 0, 0],
                    [0,  0, 0,  0, 0, 0, avar]])
    
    q_V =  Q_discrete_white_noise(2, dt = dt, var=sigV)
    Q_V = block_diag(q_V, 0., q_V, 0., 1.)

    #For const A
    F_A = np.array([[1., dt, 0.5*(dt**2),  0, 0, 0, 0],
                    [0,  1, dt,  0, 0, 0, 0],
                    [0,  0, 1,  0, 0, 0, 0],
                    [0,  0, 0,  1, dt, 0.5*(dt**2), 0],
                    [0,  0, 0,  0, 1, dt, 0],
                    [0,  0, 0,  0, 0, 1, 0],
                    [0,  0, 0,  0, 0, 0, 1]])
    
    P_A = np.array([[xvar, 0., 0,  0, 0, 0, 0],
                    [0,  100, 0,    0, 0, 0, 0],
                    [0,  0, 100,  0,    0, 0, 0],
                    [0,  0, 0,  yvar, 0, 0, 0],
                    [0,  0, 0,  0,   100, 0, 0],
                    [0,  0, 0,  0,    0, 100, 0],
                    [0,  0, 0,  0, 0, 0, avar]])
    
    q_A = Q_discrete_white_noise(3, dt = dt, var=sigA)
    Q_A = block_diag(q_A, q_A, 1.)


    R = np.array([[xvar, 0, 0],
                  [0, yvar, 0],
                  [0, 0, avar]])
    H = np.array([[1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1]]) 
    
    df_tracks_kalman, binnedCostsKalman, histBinsKalman = twoFrameLinking_withKalman(df_kp, lb, maxDiff, F_R, P_R, Q_R, R, H, 'RW', n_frames)
    
    end_kalman = timer()
    
    
    
    df_tracks_IMM, binnedCostsIMM, histBinsIMM = twoFrameLinking_withIMM(df_kp, lb, maxDiff, F_R, P_R, Q_R,
                                                  F_V, P_V, Q_V, F_A, P_A, Q_A, R, H, M, mu, n_frames)
    
    
    end_IMM = timer()
    #print(binnedCosts)
    if False:
        
        binnedCosts = [binnedCostsN, binnedCostsKalman,binnedCostsIMM]
        histBins = [histBinsN, histBinsKalman, histBinsIMM]
        titles = ['Naive', 'Kalman', 'IMM']
        for binnedCost, histBin, title in zip(binnedCosts, histBins, titles):
            costsArray = np.array(binnedCost)
            histBin = [ round(elem, 0) for elem in histBin ]
            
            plt.figure()
            plt.imshow(costsArray, cmap='hot', interpolation='nearest', aspect=16/costsArray.shape[0])
            plt.xticks(np.arange(-.5, 15.5, 1), histBin, rotation = 45)
            plt.colorbar()
            
            
            plt.xlabel('Cost Bins')
            plt.ylabel('frame')
            plt.title(title + ' Histogram of Costs')
            plt.show()
        
        print("naive time", str(end_naive - start_naive), "kalman time", str(end_kalman - end_naive))
        print("IMM time", str(end_IMM - end_kalman))
    
    return [df_tracks, df_tracks_kalman, df_tracks_IMM]
    


if __name__ == '__main__':
    df_kp = None
    im = None
    main()