import numpy as np 
import scipy.stats
import pandas as pd

class SamplePath():
    '''
    Class that generates sample paths for monte-carlo simulation of options prices
    as well as for use in hedging simulations.
    '''
    def __init__(self,dt,T,S0):
        self.dt = dt
        self.T = T
        self.S0 = S0

    def generate_paths():
        pass

class GeometricBrownianMotion(SamplePath):

    def __init__(self,mu,sigma,dt,T,S0,method = 'Euler'):
        super().__init__(dt,T,S0)
        self.mu = mu
        self.sigma = sigma
        self.method = method

    def generate_paths(self,n_paths = 1):
        num_samps = int(self.T/self.dt)
        sample_paths = np.ndarray(shape = [n_paths,num_samps])
        sample_paths[:,0] = np.array([self.S0]*n_paths)
        sample_paths[:,1:] = np.exp(self.mu*self.dt + np.sqrt(self.dt)*self.sigma*np.random.randn(n_paths,num_samps - 1))
        sample_paths = sample_paths.cumprod(axis = 1)
        return sample_paths
        
