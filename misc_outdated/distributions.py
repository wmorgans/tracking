# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 09:24:48 2020

@author: j72687wm
"""
from scipy.stats import rv_continuous
import numpy as np
from scipy.integrate import simps

class TruncatedPowerLaw(rv_continuous):
    #Rab5
    #persistant
    #μ=1.352±0.102, τ0=0.045±0.006s,  λ=1.286±0.142
    
    #Rab5
    #anti persistant
    #μ=0.518±0.004 , τ0=0.140±0.002s, λ=0.352±0.002s−1  
    
    def __init__(self, tau_0, mu, lamda, max_t, name = None):
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
    
    # def _ppf(self,x, mu, tau_0, lamda):
    #     return 1/(1-(np.exp(-lamda*x)*(tau_0/(tau_0 + x))**mu))