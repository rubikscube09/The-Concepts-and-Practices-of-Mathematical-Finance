from pathgen import GeometricBrownianMotion, SamplePath
from VanillaBS import VanillaBS
import numpy as np 

class Hedge():
    

    def __init__(self,option,mu,dt):
        self.option = option
        assert option.sigma 
        sigma = option.sigma
        T = option.T
        S0 = option.spot
        self.path = GeometricBrownianMotion(mu,sigma,dt,T,S0)
        self.portfolio = {'Equity':0,'Option':0}
        
    
class DeltaHedge(Hedge):

    def __init__(self,option,mu,dt):  
        super().__init__(option,mu,dt)

    def hedge(self):
        # Need to think about how to do this in a vectorized fashion.
        # Currently developing it with one path in mind, extension to multiple paths is easy and can be done in the future
        sample_path = self.path.generate_paths()[0]
        self.portfolio['Option'] = 1
        lst = []    
        #lst append, not fun/not a good idea
        for i in range (len(self.path.time)):
            self.option.T = self.path.time[-1] - self.path.time[i]
            self.option.spot = sample_path[i] 
            delta = self.option.compute_bs_delta()
            self.portfolio['Equity'] = delta
            portfolio_value = {'Option':self.option.compute_bs_price()*delta,'Equity' : delta*sample_path[i]}
            lst.append(portfolio_value)
        return lst, sample_path


        
