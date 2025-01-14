# Simplest Flow Matching

from torch import Tensor

class SimplestFM:
    
    def x_t(self, x_0: Tensor, x_1: Tensor, t) -> Tensor:
        return (1-t)*x_0+t*x_1
    
    def dx_t(self, x_0:Tensor, x_1:Tensor) -> Tensor:
        return x_1-x_0

# Optimal Transport Flow Matching   
class OTFM:   

    def compute_x_t(self, x_0:Tensor, x_1:Tensor, sigma_min:int) -> Tensor:
        return x_1 - (1-sigma_min)*x_0
    
    def compute_dx_t(self, x_0, x_1, sigma_min, t) -> Tensor:
        return (1-(1-sigma_min)*t)*x_0+t*x_1
    
