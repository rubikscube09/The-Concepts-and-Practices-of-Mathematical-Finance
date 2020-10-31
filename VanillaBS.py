import payoff
import numpy as np 
import scipy


class VanillaBS(): 
    def __init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None):
        self.strike = strike
        self.spot = spot
        self.T = T
        self.sigma = sigma
        self.r = r
        self.units = units
        self.price = None
    
    def bs_price():
        raise NotImplementedError()
    
    def bs_imp_vol(self,sigma_0 = None,tol = 10**(-6)):
        '''
        Newton-Raephson implied volatility for European options under Black-Scholes model assumptions.
        '''
        assert self.price, 'Need a price to calculate implied-volatility.'

        if not sigma_0 and self.sigma:
            sigma_0 = self.sigma

        
class MonteCarloBS(VanillaBS):
    def __init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None):
        super().__init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None)
    
    def bs_imp_vol(self, tol):
        super().bs_imp_vol(tol)
    
    def compute_bs_price(self):

        assert self.sigma, 'Need volatility to compute price.'


class PDEBS(VanillaBS):
    
    def __init__(self,strike):
        super().__init__()
    
