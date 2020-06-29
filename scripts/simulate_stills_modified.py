import numpy as np
import utils
import random
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, AiryDisk2DKernel
import sys
import pandas as pd
import matplotlib.animation as animation
import itertools
import utils
from pathlib import Path

class SyntheticData():
    
    def __init__(self, method, nrows, ncols, npar, tend, dt, avgrad = 0.8, paretoshape = 3, avgint = 140,
                 stdint = 50, micronstopix = 100/954.21, diff_coeff_mu = 3):
        
        self.method = method
        
        #specify image parameters
        self.nrows = nrows
        self.ncols = ncols
        self.npar = npar
        
        #time params
        self.tend = tend
        self.dt = dt
        self.tnow = 0.
        self.nframes = int(tend/dt)
        
        #Determine dist to draw particle x and y rad from
        self.avgrad = avgrad
        self.paretoshape = paretoshape
        
        #determine normal dist to draw particle intensity from
        self.avgint = avgint
        self.stdint = stdint
        
        self.micronstopix = micronstopix #micronstopix [um/pi]
        self.diff_coeff_mu = diff_coeff_mu #um^2/s
        self.diff_coeff_pix = self.d_con_mu/pow(self.micronstopix,2) #pixels^2 per sec
        
        self.store_df = pd.DataFrame(index=np.arange(self.npar*(self.nframes)),columns=['pid', 't', 'x', 'y', 'rx', 'ry', 'int', 'rot', 'v', 'frame'], dtype = float)
        self.particlesAtTime = np.zeros((self.npar,self.store_df.shape[1]))
        
        self.im = np.zeros((self.nrows, self.ncols, self.nframes),dtype='float')
        
        self.convim = self.im.copy()
        
        #make convolved image through pointspread function
        self.kernel = AiryDisk2DKernel(2)
        # gauss_kernel = Gaussian2DKernel(2)
        # tophat_kernel = Tophat2DKernel(5)
        
        self.__writeFirstFrame()
        self.__writeFrames()
        
    def displayVid(self):
        ims = []
        fig = plt.figure()
        for i in range(self.convim.shape[2]):
            if i == 0:
                plt.imshow(self.convim[:, :, i], origin = 'lower')
            im_to_plot = plt.imshow(self.convim[:, :, i], animated=True, origin = 'lower')
            ims.append([im_to_plot])
        
        ani = animation.ArtistAnimation(fig, ims, interval=1/self.dt, repeat_delay=2000, blit = True)
        
        plt.show()
        
    def writeVid(self, outDir):
        name = str(self.method) + str(self.npar) + "_" + str(round(1/np.sqrt(self.d_con_pix*2*self.dt) , 2))
        color = False
        utils.writeVid(self.convim,outDir, name, 'MJPG', 1/self.dt, color)
        self.df.to_csv(outDir/name + '.csv', index = False)
        
    def __writeFirstFrame(self):
        self.tnow = 0.
        particlecount = 0
        while particlecount < self.npar:
            x = random.uniform(int(0.2*nrows),int(0.8*self.nrows))
            y = random.uniform(int(0.2*ncols),int(0.8*self.ncols))
            radiusx = self.avgrad*(np.random.pareto(self.paretoshape)+1)
            radiusy = self.avgrad*(np.random.pareto(self.paretoshape)+1)
            intensity = np.abs(np.random.normal(loc=self.avgint,scale=self.stdint))

            rotation = random.uniform(0,2*np.pi)
            vel = random.uniform(0, 2.5)
            self.particlesAtTime[particlecount,0] = particlecount
            self.particlesAtTime[particlecount,1] = self.tnow
            self.particlesAtTime[particlecount,2] = x
            self.particlesAtTime[particlecount,3] = y
            self.particlesAtTime[particlecount,4] = radiusx
            self.particlesAtTime[particlecount,5] = radiusy
            self.particlesAtTime[particlecount,6] = intensity
            self.particlesAtTime[particlecount,7] = rotation
            self.particlesAtTime[particlecount,8] = vel
            self.particlesAtTime[particlecount,9] = 0
        
            self.im[:, :, 0] = self.__placeGaussianMat(self.im[:, :, 0], x,y,radiusx,radiusy,intensity,rotation)
            particlecount += 1
        
        self.store_df[0:self.npar] = self.particlesAtTime
        
        self.convim[:, :, 0] = convolve(self.im[:, :, 0],self.kernel)
        
        #put image into 8-bit
        self.im[:, :, 0] = np.array((255/np.amax(self.im[:, :, 0]))*self.im[:, :, 0],dtype='uint8')
        self.convim[:, :, 0] = np.array((255/np.amax(self.convim[:, :, 0]))*self.convim[:, :, 0],dtype='uint8')
        
    def __writeFrames(self):
        
        rwStep = True
        
        for frame in range(1, self.nframes):
            self.tnow += self.dt
            self.particlesAtTime[:, 1] = self.tnow
            self.particlesAtTime[:, 9] = frame
            for par in range(0,self.npar):
                
                if rwStep:
                    self.particlesAtTime[par, :] = self.__rwStep(self.particlesAtTime[par, :], self.dt)
                #self.particlesAtTime[par, :] = nearlyConstVelStep(particlesAtTime[par, :], accScale, dt)
                
                self.im[:, :, frame] = self.__placeGaussianMat(self.im[:, :, frame], self.particlesAtTime[par,2],self.particlesAtTime[par,3],self.particlesAtTime[par,4],self.particlesAtTime[par,5],self.particlesAtTime[par,6],self.particlesAtTime[par,7])

            self.store_df[frame*self.npar:(frame + 1)*self.npar] = self.particlesAtTime
                
            
            self.convim[:,:,frame] = convolve(self.im[:,:,frame],self.kernel)
            
            self.im[:,:,frame] = np.array((255/np.amax(self.im[:,:,frame]))*self.im[:,:,frame],dtype='uint8')
            self.convim[:,:,frame] = np.array((255/np.amax(self.convim[:,:,frame]))*self.convim[:,:,frame],dtype='uint8')
        
        
    def __placeGaussianMat(self, im, x_0,y_0,sig_x,sig_y,intensity,theta, nsig = 3):
        r,c = im.shape
        
        x_0_int = int(round(x_0))
        x_0_rem = x_0 - x_0_int 
        
        y_0_int = int(round(y_0))
        y_0_rem = y_0 - y_0_int 
        
        #minimum size (in pixels) in global x, y to include nsig sigma of 2d gauss in its major axis
        x_len = self.__roundUpToOdd(np.amax(np.abs([(2*nsig*sig_y+1)*np.sin(theta), (2*nsig*sig_x+1)*np.cos(theta)])))
        y_len = self.__roundUpToOdd(np.amax(np.abs([(2*nsig*sig_y+1)*np.cos(theta), (2*nsig*sig_x+1)*np.sin(theta)])))
    
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
        xmin = x_0_int - x_range
        xmax = x_0_int + x_range + 1
        ymin = y_0_int - y_range
        ymax = y_0_int + y_range + 1
        
        #make sure particle is within the bounds of the mage
        if (ymax < 0) or (xmax < 0) or (ymin > r) or (xmin > c):
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
        if xmax > c:
            mod += 'x too big'
            particle = particle[:, :-(xmax - c)]
            xmax = c
        if ymax > r:
            mod += 'y too big'
            particle = particle[:-(ymax - r), :]
            ymax = r
        
        #print(particle)
        try:
            im[ymin: ymax, xmin: xmax] += particle
        except:
            print('error')
            if mod == '':
                mod += 'no modification'
            print(mod)            
            sys.exit()
        return im

    def __rwStep(self, particle, dt):
        
        jumpscale = 1/np.sqrt(self.d_con_pix*2*dt) 
        
        theta = random.uniform(0,2*np.pi)
        dinc = random.expovariate(jumpscale)
        dx = dinc*np.sin(theta)
        dy = dinc*np.cos(theta)
        particle[2] += dx
        particle[3] += dy
    
        #could change this so that is calculates the velocity based on previos time steps
        particle[7] += random.uniform(-np.pi/4,np.pi/4)
        
        return particle
    
    def __roundUpToOdd(self, f):
        return np.ceil(f) // 2 * 2 + 1

    def __nearlyConstVelStep(particle, accScale, dt):
        accMag = random.gauss(-accScale, accScale)
        accTheta = random.gauss(0, np.pi/8)
        vel = particle[8] + accMag
        theta = particle[7] + accTheta
         
        particle[2] += (vel*1)*np.sin(theta)
        particle[3] += (vel*1)*np.cos(theta)
        
        return particle
    
    def __nearlyConstAccStep(particle, jerkScale, dt):
        return particle
        