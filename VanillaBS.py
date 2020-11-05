from types import new_class
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
        self.greeks = ['delta','gamma','theta','vega','rho']

    
    def compute_bs_greeks(self):
        '''
        All option greeks 
        '''
        assert self.sigma is not None 
        return {letter: eval('self.compute_bs_' + str(letter) + '()') for letter in self.greeks}

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
        #TODO - IMPLEMENT THIS 
        pass

    def compute_bs_rho(self):
        if self.call: 
            rho = self.strike*self.T*scipy.stats.norm.cdf(self.d2())*np.exp(-self.r*self.T)
        else:
            rho = -self.strike*self.T*scipy.stats.norm.cdf(-self.d2())*np.exp(-self.r*self.T)
    
    def d1(self):
        spot,strike,r,T,sigma = self.spot,self.strike,self.r,self.T,self.sigma
        return (np.log(spot/strike) + T*(r + 0.5*sigma**2))/(sigma*np.sqrt(T))

    def d2(self):
        return self.d1() - self.sigma*np.sqrt(self.T)

    def compute_bs_price():
        raise NotImplementedError()

class MonteCarloBS(VanillaBS):
    def __init__(self,strike,spot,T,r,sigma = None,price = None,n_samp = 10**7,call = True,seed = np.random.randint(low = 1,high = 10**8)):
        super().__init__(strike,spot,T,r,sigma,price,call)
        self.n_samp = n_samp
        self.seed = seed

    def compute_bs_price(self):
        np.random.seed(self.seed)
        print(self.seed)
        assert self.sigma is not None, 'Need volatility to compute price.'
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
        return self.sigma
    def finite_diff_delta(self,tpe = 'ctr',tol = 10**(-3),h = 10**(-4)):
        '''
        Use a forward, backward, or centered finite difference to calculate call
        delta. Uses the same path for finite differencing to avoid monte-carlo
        variance issues. Centered finite differences are preferred due to lower
        estimator bias.

        TODO - Use tolerance for estimates up to arbitrary position. 
        '''
        # Currently off from Black Scholes Delta by a factor that is O(10^(-2))
        # Monte Carlo error scales poorly so more draws maybe not worth it.
        if tpe == 'fwd':
           v1 = self.compute_bs_price()
           option2 =  MonteCarloBS(self.strike,self.spot + h, self.T,self.r,self.sigma,call=self.call, seed = self.seed) 
           v2 = option2.compute_bs_price()
           print(v2)
           print(v1)
           print(v2 - v1)
           return((v2 - v1)/h)
        elif tpe == 'ctr':
           option1 = MonteCarloBS(self.strike,self.spot - h, self.T,self.r,self.sigma,call=self.call) 
           option2 =  MonteCarloBS(self.strike,self.spot + h, self.T,self.r,self.sigma,call=self.call, seed = option1.seed) 
           v1 = option1.compute_bs_price()
           v2 = option2.compute_bs_price()
           return((v2 - v1)/(2*h))


class PDEBS(VanillaBS):
    
    def __init__(self,strike):
        super().__init__()
    
