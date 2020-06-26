import numpy as np
import random
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, AiryDisk2DKernel
import sys
import pandas as pd
import matplotlib.animation as animation
import itertools
from pathlib import Path

def roundUpToOdd(f):
    return np.ceil(f) // 2 * 2 + 1

#function that calculates the z point of a 2d Gaussian
def gaussian2d(x,y,x0,y0,sx,sy,theta,intensity, a, b, c, nsig = 3):
    
    return intensity*np.exp(-(a*pow(x-x0,2)+2*b*(x-x0)*(y-y0)+c*pow(y-y0,2)))

#function that takes image, location and radius and generates a gaussian shape in the image
def placeGaussian(im,x,y,radiusx,radiusy,intensity,rotation):
    r,c = im.shape
    Y = np.array(range(0,r))
    X = np.array(range(0,c))
    
        
    #calculate parameters of 2d gaussian
    a = pow(np.cos(rotation),2)/(2*pow(radiusx,2)) + pow(np.sin(rotation),2)/(2*pow(radiusy,2)) 
    b = -np.sin(2*rotation)/(4*pow(radiusx,2)) + np.sin(2*rotation)/(4*pow(radiusy,2))
    c = pow(np.sin(rotation),2)/(2*pow(radiusx,2)) + pow(np.cos(rotation),2)/(2*pow(radiusy,2))
    
    
    for xp in X:
        for yp in Y:
            zp = gaussian2d(xp,yp,x,y,radiusx,radiusy,rotation,intensity, a, b, c) 
            im[xp,yp] += zp
    return im

#function that takes image, location and radius and generates a gaussian shape in the image
def placeGaussianMat(im,x_0,y_0,sig_x,sig_y,intensity,theta, nsig = 3):
    r,c = im.shape
    
    x_0_int = int(round(x_0))
    x_0_rem = x_0 - x_0_int 
    
    y_0_int = int(round(y_0))
    y_0_rem = y_0 - y_0_int 
    
    #minimum size (in pixels) in global x, y to include nsig sigma of 2d gauss in its major axis
    x_len = roundUpToOdd(np.amax(np.abs([(2*nsig*sig_y+1)*np.sin(theta), (2*nsig*sig_x+1)*np.cos(theta)])))
    y_len = roundUpToOdd(np.amax(np.abs([(2*nsig*sig_y+1)*np.cos(theta), (2*nsig*sig_x+1)*np.sin(theta)])))

    x_range = int(x_len//2)
    y_range = int(y_len//2)
    
    x = np.arange(-x_range, x_range + 1)
    y = np.arange(-y_range, y_range + 1)
    
    
    
    #Make grid of values of distance (in pixels) from gaussians center
    xx, yy = np.meshgrid(x, y)
        
    #calculate parameters of 2d gaussian
    a = pow(np.cos(theta),2)/(2*pow(sig_x,2)) + pow(np.sin(theta),2)/(2*pow(sig_y,2)) 
    b = -np.sin(2*theta)/(4*pow(sig_x,2)) + np.sin(2*theta)/(4*pow(sig_y,2))
    c = pow(np.sin(theta),2)/(2*pow(sig_x,2)) + pow(np.cos(theta),2)/(2*pow(sig_y,2))
    
    #resulting discrete gaussian. x_0_rem and y_0_rem give sub pixel distance
    particle = intensity*np.exp(-((a*np.power(xx + x_0_rem,2))+(2*b*((xx + x_0_rem)*(yy + y_0_rem)))+(c*np.power(yy + y_0_rem,2))))
    
    #values to slice image with
    xmin  = pxmin = x_0_int - x_range
    xmax = pxmax = x_0_int + x_range + 1
    ymin = pymin =  y_0_int - y_range
    ymax = pymax = y_0_int + y_range + 1
    
    ppshape = particle.shape
    
    #make sure particle is within the bounds of the mage
    if (ymax < 0) or (xmax < 0) or (ymin > im.shape[0]) or (xmin > im.shape[1]):
        #print('particle does not overlap with im')
        return im
    
    mod = ''
    
    if xmin < 0:
        mod += 'x to small'
        particle = particle[:, -xmin:]
        xmin = 0
    if ymin < 0:
        mod += 'y too small'
        particle = particle[-ymin:, :]
        ymin = 0
        
    if xmax > im.shape[1]:
        mod += 'x too big'
        particle = particle[:, :-(xmax - im.shape[1])]
        xmax = im.shape[1]
    if ymax > im.shape[0]:
        mod += 'y too big'
        particle = particle[:-(ymax - im.shape[0]), :]
        ymax = im.shape[0]
    
    #print(particle)
    try:
        im[ymin: ymax, xmin: xmax] += particle
    except:
        print('error')
        
        if mod == '':
            mod += 'no modification'
        print(mod)
        imshape = im[ymin: ymax, xmin: xmax].shape
        preimshape = im[pymin: pymax, pxmin: pxmax].shape
        pshape = particle.shape
        
        print('\n y not equal')
        print(imshape[0] != pshape[0])
        
        print('\n x not equal')
        print(imshape[1] != pshape[1])
        
        print('\n slice ranges')
        print(pymin, pymax, pxmin, pxmax)
        print(ymin, ymax, xmin, xmax)
        
        print('\n imshape')
        print(preimshape)
        print(imshape)
        
        
        print('\n particle shape')
        print(ppshape)
        print(pshape)
        
        print('\n position')
        print(x_0, y_0)
        print(x_0_int, y_0_int)
        print(x_0_rem, y_0_rem)
        
        sys.exit()
    return im

def rwStep(particle, jumpscale):
    
    theta = random.uniform(0,2*np.pi)
    dinc = random.expovariate(jumpscale)
    dx = dinc*np.sin(theta)
    dy = dinc*np.cos(theta)
    particle[2] += dx
    particle[3] += dy

        # print(array2print[i,2],array2print[i,3])
    particle[7] += random.uniform(-np.pi/4,np.pi/4)
        
    return particle

def nearlyConstVelStep(particle, accScale, dt):
    accMag = random.gauss(-accScale, accScale)
    accTheta = random.gauss(0, np.pi/8)
    vel = particle[8] + accMag
    theta = particle[7] + accTheta
     
    particle[2] += (vel*1)*np.sin(theta)
    particle[3] += (vel*1)*np.cos(theta)
    
    return particle

def nearlyConstAccStep(particle, jerkScale, dt):
    return particle
    
    

#########################START MAIN

    
if __name__ == '__main__':
    
    #Specify whether to display video and whether to save video

    writeVid = True
    displayVid = False
    outdir = "..\\..\\videos\\simulated\\05062020\\RW\\"    
    
    #specify image parameters
    nrows = 300
    ncols = 300
    avgrad = 0.8
    #alpha in pareto equation
    paretoshape = 3
    npar = 20
    avgint = 140
    stdint = 50
    
    #specify temporal params
    #12
    tend = 12.
    dt = 0.012
    frames = int(tend/dt)
    
    
    micronstopix = 100/954.21 #micronstopix [um/pi]
    d_con_mu = 3 #um^2/s
    d_con_pix = d_con_mu/pow(micronstopix,2) #pixels^2 per sec
    tnow = 0.
    jumpscale = 1/np.sqrt(d_con_pix*2*dt) 
    accScale = 1.
    print('lambda = ',jumpscale)
    print('beta = ', 1/jumpscale)
    
    npars = [50]#np.arange(10, 51, 10)
    jumpscales = [jumpscale]#np.arange(0.15, 0.96, 0.1)
    #avgrads = np.arange(0.5, 2.1, 0.2)
    i = 1
    total = len(npars)*len(jumpscales)
    for (npar, jumpscale) in itertools.product(npars, jumpscales):
    
        #container for particle info at single time step
        particlesAtTime = np.zeros((npar,10))
        
        #containeer for particle info
        
        df = pd.DataFrame(index=np.arange(npar*(frames)),columns=['pid', 't', 'x', 'y', 'rx', 'ry', 'int', 'rot', 'v', 'frame'], dtype = float)
        
        # print(array2print)
        #generate blank image
        im = np.zeros((nrows,ncols, frames),dtype='float')
        convim = np.zeros((nrows,ncols, frames),dtype='float')
        # print(im)
        
        #generate initial particle positions
        particlecount = 0
        while particlecount < npar:
            # x = np.random.randint(0,nrows)
            # y = np.random.randint(0,ncols)
            x = random.uniform(int(0.2*nrows),int(0.8*nrows))
            y = random.uniform(int(0.2*ncols),int(0.8*ncols))
            # print(x,y)
            #+1 is the x_0 value
            radiusx = avgrad*(np.random.pareto(paretoshape)+1)
            radiusy = avgrad*(np.random.pareto(paretoshape)+1)
            intensity = np.abs(np.random.normal(loc=avgint,scale=stdint))
            rotation = random.uniform(0,2*np.pi)
            vel = random.uniform(0, 2.5)
            particlesAtTime[particlecount,0] = particlecount
            particlesAtTime[particlecount,1] = 0
            particlesAtTime[particlecount,2] = x
            particlesAtTime[particlecount,3] = y
            particlesAtTime[particlecount,4] = radiusx
            particlesAtTime[particlecount,5] = radiusy
            particlesAtTime[particlecount,6] = intensity
            particlesAtTime[particlecount,7] = rotation
            particlesAtTime[particlecount,8] = vel
            particlesAtTime[particlecount,9] = 0
        
            im[:, :, 0] = placeGaussianMat(im[:, :, 0],x,y,radiusx,radiusy,intensity,rotation)
            particlecount += 1
        
        df[0:particlesAtTime.shape[0]] = particlesAtTime
        
        #make convolved image through pointspread function
        gauss_kernel = Gaussian2DKernel(2)
        tophat_kernel = Tophat2DKernel(5)
        airy_kernel = AiryDisk2DKernel(2)
        convim[:, :, 0] = convolve(im[:, :, 0],airy_kernel)
        
        #put image into 8-bit
        im[:, :, 0] = np.array((255/np.amax(im[:, :, 0]))*im[:, :, 0],dtype='uint8')
        convim[:, :, 0] = np.array((255/np.amax(convim[:, :, 0]))*convim[:, :, 0],dtype='uint8')
        
        
        
        #now we start incrementing time
        t = 0
        for frame in range(1, frames):
            t += dt
            particlesAtTime[:, 1] = t
            particlesAtTime[:, 9] = frame
            for par in range(0,npar):
                
                particlesAtTime[par, :] = rwStep(particlesAtTime[par, :], jumpscale)
                particlesAtTime[par, :] = nearlyConstVelStep(particlesAtTime[par, :], accScale, dt)
                
                im[:, :, frame] = placeGaussianMat(im[:, :, frame],particlesAtTime[par,2],particlesAtTime[par,3],particlesAtTime[par,4],particlesAtTime[par,5],particlesAtTime[par,6],particlesAtTime[par,7])

            df[frame*particlesAtTime.shape[0]:(frame + 1)*particlesAtTime.shape[0]] = particlesAtTime
                
            
            convim[:,:,frame] = convolve(im[:,:,frame],airy_kernel)
            
            im[:,:,frame] = np.array((255/np.amax(im[:,:,frame]))*im[:,:,frame],dtype='uint8')
            convim[:,:,frame] = np.array((255/np.amax(convim[:,:,frame]))*convim[:,:,frame],dtype='uint8')
        
        
        
        #Display video
            #%%
            
        print(str((i/total)*100))
        i += 1
        if displayVid:
            fig = plt.figure()
            # ims is a list of lists, each row is a list of artists to draw in the
            # current frame; here we are just animating one artist, the image, in
            # each frame
            ims = []
            #for i in range(convim.shape[2]):
                #if i == 0:
                    #plt.imshow(convim[:, :, i], origin = 'lower')
                im_to_plot = plt.imshow(convim[:, :, i], animated=True, origin = 'lower')
                ims.append([im_to_plot])
            
            ani = animation.ArtistAnimation(fig, ims, interval=1/dt, repeat_delay=2000, blit = True)
    
            plt.show()
        #write to file
            
        if writeVid:
            name = str(npar) + "_" + str(round(jumpscale, 2))
            color = False
            utils.writeVid(convim,outdir, name, 'MJPG', 1/dt, color)
            df.to_csv(outdir + name + '.csv', index = False)

