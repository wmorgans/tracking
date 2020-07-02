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
from scipy.integrate import simps
from scipy.stats import rv_continuous
from scipy.stats import beta

class TruncatedPowerLaw(rv_continuous):
    #Rab5
    #persistant
    #μ=1.352±0.102, τ0=0.045±0.006s,  λ=1.286±0.142
    
        #Rab5
    #anti persistant
    #μ=0.518±0.004 , τ0=0.140±0.002s, λ=0.352±0.002s−1  
    def __init__(self, tau_0, mu, lamda, max_t):
        super().__init__()
        self.a = self.tau_0 = tau_0
        self.b = max_t
        self.mu = mu
        self.lamda = lamda

    def _pdf(self, x):
        pdf = self.nonScaled_pdf(x)
        scale = simps(pdf, x)
        print(scale)
        return (pdf/scale) 
    
    def nonScaled_pdf(self, x):
        return (self.lamda*np.exp(-self.lamda*x)*(self.tau_0/(self.tau_0 + x))**self.mu) + (np.exp(-self.lamda*x)*self.mu*((self.tau_0**self.mu)/((self.tau_0+x)**(self.mu + 1))))
    
    def _cdf(self,x):
        return (1 - (np.exp(-self.lamda*x)*(self.tau_0/(self.tau_0 + x))**self.mu))
    
class SyntheticData():
    
    def __init__(self, mode, nrows, ncols, npar, tend, dt, avgrad = 0.8, paretoshape = 3, avgint = 140,
                 stdint = 50, micronstopix = 100/954.21, diff_coeff_um = 3, acc_scale = 0.2):
        
        self.mode = mode
        
        #specify image parameters
        self.nrows = nrows
        self.ncols = ncols
        self.npar = npar
        
        #time params
        self.tend = tend
        self.dt = dt
        self.tnow = 0.
        self.nframes = int(tend/dt)
        
        #Determine dist to draw particles x and y rad from
        self.avgrad = avgrad
        self.paretoshape = paretoshape
        
        #determine normal dist to draw particle intensity from
        self.avgint = avgint
        self.stdint = stdint
        
        self.micronstopix = micronstopix #micronstopix [um/pi] #actually pix to microns
        self.diff_coeff_um = diff_coeff_um #um^2/s
        self.diff_coeff_pix = self.diff_coeff_um/pow(self.micronstopix,2) #pixels^2 per sec
        
        #var of awgn for const v
        self.accScale = acc_scale
        
        self.store_df = pd.DataFrame(index=np.arange(self.npar*(self.nframes)),columns=['pid', 't', 'x', 'y', 'rx', 'ry', 'int', 'rot', 'v', 'frame'], dtype = float)
        self.particlesAtTime = np.zeros((self.npar,self.store_df.shape[1]))
        
        self.im = np.zeros((self.nrows, self.ncols, self.nframes),dtype='float')
        self.convim = self.im.copy()
        
        #make convolved image through pointspread function
        self.kernel = AiryDisk2DKernel(2)
        # gauss_kernel = Gaussian2DKernel(2)
        # tophat_kernel = Tophat2DKernel(5)
        
        #Rab5
        #persistant
        #μ=1.352±0.102, τ0=0.045±0.006s,  λ=1.286±0.142
        # tau_0, mu, lamda, max_t
        self.persistentTime = TruncatedPowerLaw(0.045, 1.352, 1.286, 10)
        
        #Rab5
        #anti persistant
        #μ=0.518±0.004 , τ0=0.140±0.002s, λ=0.352±0.002s−1  
        self.antipersistentTime = TruncatedPowerLaw(0.14, 0.518, 0.352, 10)
        #1.337609372680772 4.617898610041352 0 6
        self.velDistParams =  {'type':'beta', 'a':1.34, 'b':4.6, 'loc':0, 'scale':6}
        self.runsAndRests = {}
        
        if self.mode == 'runsAndRests':
            self.__determineRunsAndRests()
        elif self.mode == 'RW':
            for pID in range(self.npar):
                self.runsAndRests[pID] = [('RW', self.tend + 1)]
        
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
        name = str(self.mode) + "_" + str(self.npar) + "_" + str(self.diff_coeff_um)
        color = False
        utils.writeVid(self.convim,outDir, name, 'MJPG', 1/self.dt, color)
        name += '.csv'
        self.store_df.to_csv(outDir/name, index = False)
        
    def __determineRunsAndRests(self):
        
        for pID in range(self.npar):
            self.runsAndRests[pID] = []
            if 0.9 > np.random.uniform():
                mode = 'RW'
                #Determine duration from truncated powerlaw 
                duration = self.antipersistentTime.rvs(1) 
            else:
                mode = 'Dir'
                #Determine duration from truncated powerlaw
                duration =  self.persistentTime.rvs(1)         
            self.runsAndRests[pID].append((mode, duration))
            
        persist = True
        i = 0
        while persist:
            persist = False
            for pID in range(self.npar):
                if self.runsAndRests[pID][-1][1] >= self.tend:
                    continue
                else:
                    persist = True
                    if self.runsAndRests[pID][-1][0] == 'RW':
                        mode = 'Dir'
                        #Determine duration from truncated powerlaw 
                        duration = self.persistentTime.rvs(1)
                    if self.runsAndRests[pID][-1][0] == 'Dir':
                        mode = 'RW'
                        #Determine duration from truncated powerlaw 
                        duration = self.antipersistentTime.rvs(1)
                    time = self.runsAndRests[pID][-1][1] + duration
                    self.runsAndRests[pID].append((mode, time))
                    
        self.tempRunsAndRests = self.runsAndRests.copy()
                    
            
        
    def __writeFirstFrame(self):
        self.tnow = 0.
        particlecount = 0
        while particlecount < self.npar:
            x = random.uniform(int(0.2*self.nrows),int(0.8*self.nrows))
            y = random.uniform(int(0.2*self.ncols),int(0.8*self.ncols))
            radiusx = self.avgrad*(np.random.pareto(self.paretoshape)+1)
            radiusy = self.avgrad*(np.random.pareto(self.paretoshape)+1)
            intensity = np.abs(np.random.normal(loc=self.avgint,scale=self.stdint))

            rotation = random.uniform(0,2*np.pi)
            #vel = random.uniform(0, 2.5)
            self.particlesAtTime[particlecount,0] = particlecount
            self.particlesAtTime[particlecount,1] = self.tnow
            self.particlesAtTime[particlecount,2] = x
            self.particlesAtTime[particlecount,3] = y
            self.particlesAtTime[particlecount,4] = radiusx
            self.particlesAtTime[particlecount,5] = radiusy
            self.particlesAtTime[particlecount,6] = intensity
            self.particlesAtTime[particlecount,7] = rotation
            self.particlesAtTime[particlecount,8] = np.nan
            self.particlesAtTime[particlecount,9] = 0
        
            self.im[:, :, 0] = self.__placeGaussianMat(self.im[:, :, 0], x,y,radiusx,radiusy,intensity,rotation)
            particlecount += 1
        
        self.store_df[0:self.npar] = self.particlesAtTime
        
        self.convim[:, :, 0] = convolve(self.im[:, :, 0],self.kernel)
        
        #put image into 8-bit
        self.im[:, :, 0] = np.array((255/np.amax(self.im[:, :, 0]))*self.im[:, :, 0],dtype='uint8')
        self.convim[:, :, 0] = np.array((255/np.amax(self.convim[:, :, 0]))*self.convim[:, :, 0],dtype='uint8')
        
    def __writeFrames(self):
        
        for frame in range(1, self.nframes):
            self.tnow += self.dt
            self.particlesAtTime[:, 1] = self.tnow
            self.particlesAtTime[:, 9] = frame
            
            for par in range(0, self.npar):
                
                t = self.tnow - self.dt
                t_step = self.dt
                
                while t < self.tnow:
                    
                    
                    toPop = False    
                    t = self.tempRunsAndRests[par][0][1]
                    
                    if t > self.tnow:
                        dt = t_step
                    elif (t < self.tnow) and (t > self.tnow - t_step):
                        dt = t - (self.tnow - self.dt)
                        t_step -= dt
                        toPop = True
                    else:
                        print('error')
                    
                    if self.tempRunsAndRests[par][0][0] == 'RW':
                        
                        self.particlesAtTime[par, :] = self.__rwStep(self.particlesAtTime[par, :], dt)
                        
                    elif self.tempRunsAndRests[par][0][0] == 'Dir':
                        self.particlesAtTime[par, :] = self.__nearlyConstVelStep(self.particlesAtTime[par, :], dt)
                        
                        
                    if toPop:
                        self.tempRunsAndRests[par].pop(0)
                        
                    
                    
                
                self.im[:, :, frame] = self.__placeGaussianMat(self.im[:, :, frame], self.particlesAtTime[par,2],self.particlesAtTime[par,3],self.particlesAtTime[par,4],self.particlesAtTime[par,5],self.particlesAtTime[par,6],self.particlesAtTime[par,7])

            self.store_df[frame*self.npar:(frame + 1)*self.npar] = self.particlesAtTime
                
            
            self.convim[:,:,frame] = convolve(self.im[:,:,frame],self.kernel)
            
            self.im[:,:,frame] = np.array((255/np.amax(self.im[:,:,frame]))*self.im[:,:,frame],dtype='uint8')
            self.convim[:,:,frame] = np.array((255/np.amax(self.convim[:,:,frame]))*self.convim[:,:,frame],dtype='uint8')
        
        
    def __placeGaussianMat(self, im, x_0,y_0,sig_x,sig_y,intensity,theta, nsig = 3):
    
        row,col = im.shape
        
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
        if (ymax < 0) or (xmax < 0) or (ymin > row) or (xmin > col):
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
        if xmax > col:
            mod += 'x too big'
            particle = particle[:, :-(xmax - col)]
            xmax = col
        if ymax > row:
            mod += 'y too big'
            particle = particle[:-(ymax - row), :]
            ymax = row
        
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
        jumpscale = 1/np.sqrt(self.diff_coeff_pix*2*dt) 
        
        theta = random.uniform(0,2*np.pi)
        dinc = random.expovariate(jumpscale)
        dx = dinc*np.sin(theta)
        dy = dinc*np.cos(theta)

        particle[2] += dx
        particle[3] += dy
    
        #could change this so that is calculates the velocity based on previos time steps
        particle[7] += random.uniform(-np.pi/4,np.pi/4)
        particle[8] = np.nan
        
        return particle
    
    def __roundUpToOdd(self, f):
        return np.ceil(f) // 2 * 2 + 1

    def __nearlyConstVelStep(self, particle, dt):
        if np.isnan(particle[8]):
            vel = beta.rvs(a = self.velDistParams['a'], b = self.velDistParams['b'], loc = self.velDistParams['loc'], scale = self.velDistParams['scale'], size=1)
            velPix = vel/self.micronstopix
            particle[8] = velPix
        
        
        velPix = particle[8]
        accMag = random.gauss(0, self.accScale)
        accTheta = random.gauss(0, np.pi/8)
        velPix += accMag
        theta = particle[7] + accTheta
         
        particle[2] += (velPix*1)*np.sin(theta)
        particle[3] += (velPix*1)*np.cos(theta)
        
        return particle
    
    def __nearlyConstAccStep(self, particle, jerkScale, dt):
        return particle
        