# Simplest Flow Matching

from torch import Tensor
import torch

class SimplestFM:
    
    def compute_x_t(self, x: Tensor, x_1: Tensor, t) -> Tensor:
        return (1-t)*x+t*x_1
    
    def compute_dx_t(self, x:Tensor, x_1:Tensor) -> Tensor:
        return x_1-x
    
# Diffusion conditional Flow Matching
class DiffusionFM:
    
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def T_t(self, t: torch.Tensor):
        '''
        Computes the integral of beta_t with respect to t
        where beta_t is linear - beta_t = beta_min + t * (beta_max - beta_min)
        '''
        return self.beta_min*t + 0.5*torch.pow(t, 2)*(self.beta_max-self.beta_min)
    
    def T_dt(self, t: torch.Tensor):
        '''
        T_dt is the same as beta in the artible Lipman et. al 2023
        '''
        return self.beta_min + t*(self.beta_max-self.beta_min) 
    
    def alpha_t(self, t: torch.Tensor):
        return torch.exp((-0.5)*self.T_t(t))
    
    def sigma_t(self, t: torch.Tensor):
        return torch.sqrt(1-torch.pow(self.alpha_t(1-t), 2))
    
    def mu_t(self, t: torch.Tensor, x1: torch.Tensor):
        return self.alpha_t(1-t)*x1
    
    def compute_x_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        return self.sigma_t(t)*x + self.mu_t(t, x_1)       
    
    def compute_dx_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        numerator = torch.exp(-self.T_t(1-t))*x - torch.exp((-0.5)*self.T_t(1-t))*x_1
        numerator = torch.exp(-self.T_t(1. - t)) * x - torch.exp(-0.5 * self.T_t(1.-t))* x_1
        denominator = 1-torch.exp(-self.T_t(1-t))
        return -0.5*self.T_t(1-t)*(numerator/denominator)

# Flow matching with optimal transport
# COTFM - Conditional Optimal Transport Flow Matching
class COTFM:   

    def compute_x_t(self, x:Tensor, x_1:Tensor, sigma_min, t) -> Tensor:
        return (1-(1-sigma_min)*t)*x + t*x_1
    
    def compute_dx_t(self, x, x_1, sigma_min) -> Tensor:
        return x_1 - (1 - sigma_min)*x
    
