from payoff import VanillaPayoff
import numpy as np 
import scipy.stats


class VanillaBS(): 

    def __init__(self,strike,spot,T,r,sigma = None,price = None, call = True):
        self.strike = strike
        self.spot = spot
        self.T = T
        self.sigma = sigma
        self.r = r
        self.price = price
        self.call = call

    
    def compute_bs_greeks():
        pass

    def compute_bs_delta(self):
        '''
        Compute Delta of European Option
        '''
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

    def compute_bs_rho(self):
        if self.call: 
            rho = self.strike*self.T*scipy.stats.norm.cdf(self.d2())*np.exp(-self.r*self.T)
        else:
            rho = -self.strike*self.T*scipy.stats.norm.cdf(-self.d2())*np.exp(-self.r*self.T)
    
    def d1(self):
        spot,strike,r,T,sigma = self.spot,self.strike,self.r,self.T,self.sigma
        print('chnged')
        return (np.log(spot/strike) + T*(r + 0.5*sigma**2))/(sigma*np.sqrt(T))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)

    def bs_price():
        raise NotImplementedError()

class MonteCarloBS(VanillaBS):
    def __init__(self,strike,spot,T,r,sigma = None,price = None,n_samp = 10**7,call = True):
        super().__init__(strike,spot,T,r,sigma,price,call)
        self.n_samp = n_samp
    
    def compute_bs_imp_vol(self, tol):
        super().bs_imp_vol(tol)
    
    def compute_bs_price(self):

        assert self.sigma is not None, 'Need volatility to compute price.'
        print()
        return np.mean(VanillaPayoff(self.strike,self.call).compute_payoff(self.spot*np.exp((self.r - self.sigma**2/2)*self.T + self.sigma*np.sqrt(self.T)*np.random.normal(size = self.n_samp))))
 
    def compute_bs_imp_vol(self,tol = 10**(-5)):
        '''
        Newton-Raephson implied volatility for European options under Black-Scholes model assumptions.
        '''
        assert self.price, 'Need a reference price to calculate implied-volatility.'
        self.sigma = 0.5
        sigma_prev = 0
        while abs(sigma_prev - self.sigma) > tol:
            sigma_prev = self.sigma
            vega = self.compute_bs_vega()
            bs_price = self.compute_bs_price()
            self.sigma = self.sigma - (bs_price - self.price)/vega
            print(self.sigma)
        return self.sigma

class PDEBS(VanillaBS):
    
    def __init__(self,strike):
        super().__init__()
    
