from pathgen import SamplePath
import numpy as np 

class Hedge():
    

    def __init__(self,path : SamplePath, option:  VanillaBS):
        self.path = path
        self.option = option
        self.portfolio = {'Equity':0,'Option':0}
        
    
class DeltaHedge(Hedge):

    def __init__(path,option):  
        super().__init__(path,option)

    def hedge(self):
        # Need to think about how to do this in a vectorized fashion.
        
