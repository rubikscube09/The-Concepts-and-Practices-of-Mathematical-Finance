class Payoff():
    
    def __init__(self,strike,spot):
        self.strike = strike
    
    def compute_payoff(self,spot):
        raise NotImplementedError()

class VanillaPayoff(Payoff):    
    def __init__(self,strike,call_or_put):
        super().__init__(strike)
        self.call = call_or_put
    
    def compute_payoff(self,spot):
        if self.call:
            return max(spot - self.strike,0)
        else:
            return max(self.strike - spot,0)
        