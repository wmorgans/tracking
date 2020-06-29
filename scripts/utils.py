# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:30:52 2020

@author: j72687wm
"""
import cv2 as cv
import numpy as np
import pandas as pd
import colorsys
import os
import exifread
import datetime
from pathlib import Path


def readVid(fileDir, singleFile):
    src = []
    retvals = []
    if singleFile:    
        for f_name in filter(lambda fileInDir: fileInDir.endswith('.tif'), os.listdir(fileDir)):
            retval, mats = cv.imreadmulti(fileDir + f_name, flags = -1)
            retvals.append(retval)
            src.append(np.dstack(mats).astype('float32'))
    else:
        
        for folder in filter(lambda itemInDir: os.path.isdir(fileDir + itemInDir), os.listdir(fileDir)):
            vid = []
            files = [ x for x in os.listdir(fileDir + folder) if x.endswith('.tif')]
            index_end = [file.find('.') for file in files]
            index_start = []
            
            for file, index in zip(files, index_end):
                i = 1
                while(True):
                    i_start = index -i
                    if not file[i_start].isdigit():
                        index_start.append(int(i_start + 1))
                        break
                    i += 1
            print(len(files))
            frameNums = [int(files[i][index_start[i]:index_end[i]]) for i in range(len(files))]
            orderedFiles = [x for _,x in sorted(zip(frameNums,files))]
            
            for f_name in orderedFiles:
                vid.append(cv.imread(fileDir + folder + "\\" + f_name, flags = -1))
            src.append(np.dstack(vid).astype('float32'))
        
    print('videos loaded')        
        
    return src

#Convert csv from form of synthetic data to dataframe used by stitching method
def loadCSV(fileLoc):
    
    df_load = pd.read_csv(fileLoc, index_col=False, header=0)
    df = pd.concat([df_load['x'], df_load['x'], df_load['y'], df_load['t'], df_load['t'], df_load['pid'], df_load['t']], axis = 1)
    df.columns = ['pointID', 'x', 'y', 'timePoint', 'a', 'trackID', 't']
    df['a'] = np.pi*df_load['rx']*df_load['ry']
    df['pointID'] = df.index.values
    
    #optional columns
    if 'v' in df_load.columns:
        df['v'] = df_load['v']
    if 'frame' in df_load.columns:
        df['timePoint'] = df_load['frame']
    if 'z' in df_load.columns:
        df['z'] = df_load['z']
    else:
        df['z'] = 0
        
    #make sure columns which should be int are int
    df[['pointID', 'trackID', 'timePoint']] = df[['pointID', 'trackID', 'timePoint']].astype('int')    
    return df

# <?xml version="1.0" encoding="UTF-8" standalone="no"?>
# <root>
# <TrackContestISBI2012 snr="2" density="low" scenario="virus"> <!-- Indicates the data set -->
# <particle> <!-- Definition of a particle track -->
# <detection t="4" x="14" y="265" z="5.1"> <!-- Definition of a track point -->
# <detection t="5" x="14.156" y="266.5" z="4.9">
# <detection t="6" x="15.32" y="270.1" z="5.05">
# </particle>
# <particle> <!-- Definition of another particle track -->
# <detection t="14" x="210.14" y="12.5" z="1">
# <detection t="15" x="210.09" y="13.458" z="1.05">
# <detection t="16" x="210.19" y="14.159" z="1.122">
# </particle>
# </TrackContestISBI2012>
# </root>
def writeXml(dF, fileLoc=None, mode='w'):
    #write 1 track at a time
    #write each row with df slice
    def row_to_xml(df_row):
        xml_row = '<detection>'
        if 'z' in df_row.index:
            z = df_row['z']
        else:
            z = 0.
                
        xml_row += 't="{:.0f}" x="{:.2f}" y="{:.2f}" z="{:.2f}"'.format(df_row['timePoint'], df_row['x'], df_row['y'], z)
        xml_row += '</detection>'
        return xml_row
    def track_to_xml(track):
        res = '<particle>\n'
        res += '\n'.join(track.apply(row_to_xml, axis=1))
        res += '\n</particle>'
        return res

    df = dF.copy()
    result = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    result += '\n<root>'
    #change this to be info about groudtruthData
    result += '\n<TrackContestISBI2012>\n'
    result += '\n'.join([track_to_xml(df[df['trackID'] == track]) for track in pd.unique(df['trackID'])])
    
    result += '\n</TrackContestISBI2012>'
    result += '\n</root>'

    if fileLoc is None:
        return result
    with open(fileLoc, mode) as f:
        f.write(result)

def getNColours(N):
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    LoT =  list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
    rgb = [(t[0]*255, t[1]*255, t[2]*255) for t in LoT]
    return rgb

def writeVid(video,outdir, name, codec, fps, color):
    vid = floatToUint(video)
    vidShape = vid.shape
    fourcc = cv.VideoWriter_fourcc(codec[0], codec[1], codec[2], codec[3])
    filePath = outdir /name

    videoWriter = cv.VideoWriter(str(filePath), fourcc, fps, (vidShape[1],vidShape[0]), isColor = color)
    for t in range(vidShape[-1]):
        if color:
            frame = vid[:,:,:,t]
        else:
            frame = vid[:,:,t]
        videoWriter.write(frame)
        
def minMaxNorm(im, cut, scale):
    img = im.copy()
    #either shift pixel intesities to start from 0 or set negative values to 0.
    if not cut:
        img = img - np.amin(img, axis = (0, 1))   
    else:
        img = img.clip(min=0)   
    #scale values to span user defined range 
    img = (scale/np.amax(img, axis = (0, 1)))*np.array(img,dtype='float32')
    return img

def floatToUint(arF):
    return minMaxNorm(arF, False, 255.0).astype('uint8')

def viewVideo(vids, titles, cmap):
    
    t = 0
    vids = [floatToUint(vid) for vid in vids]
    
    frames = vids[0].shape[-1]
    while t < frames:
        
        for i in range(len(vids)):
            if cmap[i]:
                if len(vids[i].shape) == 3:
                    im_color = cv.applyColorMap(floatToUint(vids[i][:,:,t]), cv.COLORMAP_JET)
                elif len(vids[i].shape) == 4:
                    im_color = cv.applyColorMap(floatToUint(vids[i][:,:,:,t]), cv.COLORMAP_JET)
                else:
                    print('something is awry cmap')
                    return
                cv.imshow(titles[i], im_color)
            else:
                if len(vids[i].shape) == 4:
                    cv.imshow(titles[i], vids[i][:,:,:,t])
                elif len(vids[i].shape) == 3:
                    cv.imshow(titles[i], vids[i][:,:,t])
                else:
                    print('something is awry')
                    return
                
        k = cv.waitKey(33)
        if k == ord('q'):
            break
        elif k == ord('a'):
            if t <= 0: #break if step to negative frames
                break
            else:
                t -= 1 # stepback
        elif k == ord('d'):
            t += 1 # step forward
        elif k == ord('z'):
            if t <= 4:
                break
            else:
                t -= 5 # stepback by 5
        elif k == ord('c'):
            t += 5 # step forward by 5
    cv.destroyAllWindows()
    
def writeTracksOnIm(vid, df):
    
    imShape = vid.shape
    rgb = getNColours(len(pd.unique(df.trackID.dropna())))
    
    rgb = dict(zip(pd.unique(df.trackID.dropna()), rgb))

    base = np.ones((imShape[0],imShape[1],3,imShape[2]))
    
    for t in range(imShape[2]):
        frame = cv.cvtColor(floatToUint(vid[:,:,t]),cv.COLOR_GRAY2RGB)        
        dfT = df.loc[(df['timePoint'] <= t) & (df['timePoint'] >= t -20),:].dropna()
        
        dfTint = dfT.astype({'x': 'int','y': 'int', 'trackID': 'int' })
        
        for row in dfTint.itertuples(index=False):
            frame = cv.circle(frame,(row[1],row[2]), 2, rgb[row[5]],thickness = -1)
        
        base[:,:,:,t] = frame
    return base
  
#written by dan, datetime_store = find_datetime_tif(filenames)
#Use timescales = np.diff(datetime_store) to get time steps
def find_datetime_tif(filenames):
    datetime_store = []
    for i in range(0,len(filenames)):
        # print(filenames[i])    
        f = open(filenames[i], 'rb')
        tags = exifread.process_file(f)
        data = str(tags.get("Image DateTime"))
        # print(data)
        realtime = datetime.strptime(data, '%Y%m%d %H:%M:%S.%f')
        secondtemp = realtime.second + realtime.microsecond/1e6
        datetime_store.append(secondtemp)
    return datetime_store