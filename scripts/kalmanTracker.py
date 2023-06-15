import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm, det, inv

inff = 1000000000

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

def calcCostMatrixKalman(window, tracks, pointRows, kalmanDict, x):

    costMatrix = np.zeros((len(tracks), len(pointRows)))
    
    trackVector = np.zeros((len(tracks), 1, len(x)))
    
    i = 0
    for track in tracks:
        trackVector[i, 0, :] = np.dot(kalmanDict[track].H, kalmanDict[track].X).T
        i += 1
    
    pointVector = np.zeros((1, len(pointRows), len(x)))
    pointVector[0, :, :] = window.loc[pointRows,x]
    
    costMatrix = np.sqrt(np.sum((trackVector - pointVector)**2, axis = 2))
    
    return costMatrix


def twoFrameLinking_withKalman(dF, n_frames, lb, maxDiff, F, P, Q, R, H, name, x):
    df = dF.copy()
    binnedCosts = []
    halfway = False
    pred_x = []
    pred_y = []
    for currentFrame in range(n_frames):
        
        if (round(100*(currentFrame/n_frames)) == 50) & (halfway == False):
            halfway = True
            print('halfway')
            
        if currentFrame == 0:
            #intilaise track ID. (add 1 so starts from 1)
            df.loc[df['frame'] == 0, 'trackID'] = df.loc[df['frame'] == 0, 'pointID'] + 1
            kalmanDict = {}
            
            for track in pd.unique(df.loc[df['frame'] == 0, 'trackID']):
                y = df.loc[((df['frame'] == 0) & (df['trackID'] == track)), x].values.ravel()
                X_o = y.dot(H).T
                              
                #make a new calman filter object and store within a dictionary using track ID as key
                kalmanDict[track] = KalmanFilter(X_o, F, Q, R, P, H, name + "_" + str(track))
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
                
                for track in newTracks:
                    y = df.loc[df['trackID'] == track, x].values.ravel()
                    X_o = y.dot(H).T
                                      
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
        window = df.loc[(df['frame'] >= startFrame) & (df['frame'] <= currentFrame),:]
        
        pointIDs = df.loc[df['frame'] == currentFrame].index
        
        if len(pointIDs) < 1:
            print('no observations at current time step')
            continue
        
        costMatrix = calcCostMatrixKalman(window,activeTracks, pointIDs, kalmanDict, x)       
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
            y = df.loc[df['trackID'] == track, x].values.ravel()
            X_o = y.dot(H).T
                              
            #make a new kalman filter object and store within a dictionary using track ID as key
            kalmanDict[track] = KalmanFilter(X_o, F, Q, R, P, H, name + "_" + str(track))
        
        if len(assignedAndBelowThresh) > 0:
            #update existing kalman filters with assigned readings
            for track, point in zip(activeTracks[tracks], pointIDs[points]):
                y = df.loc[point, x].values.ravel()
                y = np.array([[y[0], y[1], y[2], y[3]]]).T
                kalmanDict[track].update(y)
        
    #remove tracks with fewer than 20 occurances    
    #dfTrackIDs = df[['pointID', 'trackID']].groupby('trackID').count()
    #tracksToRemove = dfTrackIDs.loc[dfTrackIDs['pointID'] < 20, :]
    #df_excessTracksRemoved = df.copy()
    #df_excessTracksRemoved.loc[df_excessTracksRemoved['trackID'].isin(tracksToRemove.index), 'trackID'] = np.nan
    return df
	
def main(df_kp, n_frames, lb, maxDiff, F, P, Q, R, H, mode, x ):
    
    if (df_kp is None or n_frames is None):
        return None
    
    df_tracks_kalman = twoFrameLinking_withKalman(df_kp, n_frames, lb, maxDiff, F, P, Q, R, H, mode, x)
    
    return df_tracks_kalman
    


if __name__ == '__main__':
    df_kp = None
    im = None
    main()