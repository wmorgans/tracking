# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:39:04 2020

@author: j72687wm
"""
import numpy as np
import cv2 as cv
import utils
import pandas as pd
import statistics as stat
from timeit import default_timer as timer

#https://stackoverflow.com/questions/42516569/numpy-add-variable-number-of-dimensions-to-an-array
def atleast_kd(array, k):
    array = np.asarray(array)
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)

def subtractMeanTime(im):
    img = im.copy()
    img = img.astype('float32')
    n_dim = img.ndim
    
    bg = atleast_kd(np.mean(img, axis = tuple(range(n_dim)[2:])), n_dim)
    
    return img - bg

def LoG_filter(im):
    imShape = im.shape
    gaus = np.empty_like(im)
    LoG = gaus.copy()
    
    for t in range(imShape[2]):
        gaus[:,:,t] = utils.minMaxNorm(cv.GaussianBlur(im[:,:,t],(5,5),1.5), False, 1.0)
        LoG[:,:,t] = cv.Laplacian(gaus[:,:,t], ddepth = cv.CV_32F, ksize=1)
        LoG[:,:,t] = utils.minMaxNorm(LoG[:,:,t]*-1, True, 1.0)
    return LoG

def binariseIm(im):
    imShape = im.shape
    adapThresh = np.empty_like(im)
    for t in range(imShape[2]):
        adapThresh[:,:,t] = cv.adaptiveThreshold(utils.floatToUint(im[:,:,t]), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -17)
    return adapThresh

def detectBlobs(im):
    imShape = im.shape
    params = cv.SimpleBlobDetector_Params()
    im = utils.floatToUint(im)      
    # Disable unwanted filter criteria params
    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByColor = False
    params.filterByCircularity = False
    params.filterByArea = True
    
    #Set filter params (Should make this paramerterisable)
    params.minArea = 2
    params.maxArea = 50
    params.thresholdStep = 2
    params.minThreshold = 254
    params.maxThreshold = 255
    params.minRepeatability = 1
    params.minDistBetweenBlobs = 2
    
    detector = cv.SimpleBlobDetector_create(params)
    kpss = []
    nkp = []
    
    for t in range(imShape[2]):
        kps = detector.detect(im[:,:,t])
        kpss.append(kps)
        nkp.append(len(kps))
    
    return [kpss, nkp]

def dataFrameOfBlobs(kpss, nkp):
    #Convert list of lists of kp objects to dataframe
    df = pd.DataFrame(index=np.arange(sum(nkp)),columns=['pointID', 'x', 'y', 'timePoint', 'a'], dtype = float)
    
    #may need to change this to start at 1
    pointID = 0
    frame = 0
    
    for frame_kps in kpss:
        for kp in frame_kps:
            df.loc[pointID] = [pointID, kp.pt[0], kp.pt[1], frame, kp.size]
            pointID += 1
        frame += 1 
        
    return df

def main(im):
    
    if im is None:
        return
    #Background sub/noise reduction
    start = timer()
    im_back_sub = subtractMeanTime(im)
    #Process (make features standout from background)
    im_LoG = LoG_filter(im_back_sub)
 
    im_bin = binariseIm(im_LoG)
 
    #Blob detect 
    kpss, nkp = detectBlobs(im_bin)
    #returns dataframe of keypoints
    df_kp = dataFrameOfBlobs(kpss, nkp)    
    end = timer()

    print("time to detect objects", str(end - start))    
    
    print("Number of detections: ", str(len(df_kp.index)))
    print("Number of detections per frame: ", str(len(df_kp.index)/1000), str(stat.mean(nkp)))
    print("std: +/-", str(stat.stdev(nkp)))
    print("min detections:", str(min(nkp)), "max detections:", str(max(nkp)))
    return df_kp


if __name__ == '__main__':
    im = None
    main()
    
