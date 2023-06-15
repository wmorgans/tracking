# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:41:58 2020

@author: j72687wm
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

inff = 1000000000

def calcCostMatrix(window, trackRows, pointRows):

    costMatrix = np.zeros((len(trackRows), len(pointRows)))
    trackVector = np.zeros((len(trackRows), 1, 4))
    trackVector[:, 0, :] = window.loc[trackRows,['x', 'y', 'a', 'i']]
    
    
    pointVector = np.zeros((1, len(pointRows), 4))
    pointVector[0, :, :] = window.loc[pointRows,['x', 'y', 'a', 'i']]
    
    
    costMatrix = np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2))
    
    return costMatrix
	
def twoFrameLinking(dF, lb, maxDiff, n_frames):
    df = dF.copy()

    halfway = False
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['frame'] == 0, 'trackID'] = df.loc[df['frame'] == 0, 'pointID'] + 1
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
        
        #print(mostRecentTracks)
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
                continue
            
        activeTracks, trackIDs = zip(*mostRecentTracks)
        trackIDs = np.asarray(trackIDs)
        activeTracks = np.asarray(activeTracks)
        
        #update window to include current frame
        window = df.loc[(df['frame'] >= startFrame) & (df['frame'] <= currentFrame),:]
        pointIDs = df.loc[df['frame'] == currentFrame].index
        
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
        
    return df
	
def main(df_kp, n_frames, lb = 3, maxDiff = 50):
    
    if (df_kp is None or n_frames is None):
        return None
    
    df_tracks = twoFrameLinking(df_kp, lb, maxDiff, n_frames)

    return df_tracks
	
if __name__ == '__main__':
    df_kp = None
    im = None
    main()

