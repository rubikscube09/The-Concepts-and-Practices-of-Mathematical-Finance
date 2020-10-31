import payoff
import numpy as np 
import scipy.stats


class VanillaBS(): 

    def __init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None, call = True):
        self.strike = strike
        self.spot = spot
        self.T = T
        self.sigma = sigma
        self.r = r
        self.units = units
        self.price = price
        self.call = call

    
    def compute_bs_greeks():
        pass

    def compute_bs_delta(self):
        if self.call: 
            return scipy.stats.norm.cdf(self.d1())
        else:
            return 1 - scipy.stats.norm.cdf(self.d1())

    def compute_bs_gamma(self):
        return (scipy.stats.norm.pdf(self.d1()))/(self.spot*self.sigma*np.sqrt(self.T))
    def compute_bs_vega(self):
        return self.spot*scipy.stats.norm.pdf(self.d1())*np.sqrt(self.T)
    def compute_bs_theta():
        pass    

    def compute_bs_rho():
        pass
    
    def d1(self):
        spot,strike,r,T,sigma = self.spot,self.strike,self.r,self.T,self.sigma
        return (np.log(spot/strike) + T*(r + sigma**2))/(sigma*np.sqrt(T))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)

    def bs_price():
        raise NotImplementedError()
    
    def bs_imp_vol(self,sigma_0 = None,tol = 10**(-6)):
        '''
        Newton-Raephson implied volatility for European options under Black-Scholes model assumptions.
        '''
        assert self.price, 'Need a price to calculate implied-volatility.'

        if not sigma_0 and self.sigma:
            sigma_0 = self.sigma
        else:
            sigma_0 = 0.2
            self.sigma = sigma_0
        # TODO - replace python loop w/ numpy loop 
        pass
        
class MonteCarloBS(VanillaBS):
    def __init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None,n_samp = 1000000):
        super().__init__(self,strike,spot,T,r,sigma = None,units = 'y',price = None)
    
    def bs_imp_vol(self, tol):
        super().bs_imp_vol(tol)
    
    def compute_bs_price(self):

        assert self.sigma, 'Need volatility to compute price.'
        return np.mean(self.spot*np.exp((self.r - self.sigma**2/2)*self.T + self.sigma*np.random.normal(size = self.n_samp)))


class PDEBS(VanillaBS):
    
    def __init__(self,strike):
        super().__init__()
    
