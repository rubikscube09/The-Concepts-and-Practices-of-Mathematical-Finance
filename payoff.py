class Payoff():
    
    def __init__(self,strike):
        self.strike = strike
    
    def compute_payoff():
        raise NotImplementedError()

class VanillaPayoff(Payoff):    
    def __init__(self,strike,call_or_put):
        super().__init__(strike)
        self.call = call_or_put
    
    def compute_payoff(self,spot):
        if self.call:
            a = spot - self.strike
            a[a < 0] = 0
        else:
            a = self.strike - spot
            a[a < 0] = 0
        return a
            
        