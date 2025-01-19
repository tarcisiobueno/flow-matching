# Simplest Flow Matching

import torch

class SimplestFM:
    
    def compute_x_t(self, x: torch.Tensor, x_1: torch.Tensor, t) -> torch.Tensor:
        return (1-t)*x+t*x_1
    
    def compute_dx_t(self, x:torch.Tensor, x_1:torch.Tensor) -> torch.Tensor:
        return x_1-x
    
class DiffusionFM:
    
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = 1e-5

    def T_t(self, t: torch.Tensor):
        '''
        Computes the integral of beta_t with respect to t
        where beta_t is linear - beta_t = beta_min + t * (beta_max - beta_min)
        '''    
        return self.beta_min*t + 0.5* (t**2) *(self.beta_max-self.beta_min)
    
    def T_dt(self, t: torch.Tensor):
        '''
        T_dt is the same as beta in the artible Lipman et. al 2023
        '''
        return self.beta_min + t*(self.beta_max-self.beta_min) 
    
    def alpha_t(self, t: torch.Tensor):
        return torch.exp(-0.5*self.T_t(t))
    
    def sigma_t(self, t: torch.Tensor):
        return torch.sqrt(1.- self.alpha_t(1. - t) ** 2)
    
    def mu_t(self, t: torch.Tensor, x1: torch.Tensor):
        return self.alpha_t(1.-t)*x1
    
    def compute_x_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        return self.sigma_t(t)*x + self.mu_t(t, x_1)       
    
    def compute_dx_t(self, x: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor):
        numerator = torch.exp(-self.T_t(1-t))*x - torch.exp((-0.5)*self.T_t(1-t))*x_1
       
        denominator = 1-torch.exp(-self.T_t(1-t))
   
        return -0.5*self.T_dt(1-t)*(numerator/denominator)

    def loss(self, model: torch.nn, x_1: torch.Tensor):
        
        x_0 = torch.randn_like(x_1, dtype=x_1.dtype, device=x_1.device)       
        t = torch.rand(len(x_1), 1, dtype=x_1.dtype, device=x_1.device) * (1-self.eps)
        x_t = self.compute_x_t(x_0, x_1, t)
        dx_t = self.compute_dx_t(x_t, x_1, t)
        v_pred = model(x_t, t)      
          
        return torch.mean((v_pred - dx_t)**2)

# Flow matching with optimal transport
# COTFM - Conditional Optimal Transport Flow Matching
class COTFM:   

    def compute_x_t(self, x:torch.Tensor, x_1:torch.Tensor, sigma_min, t) -> torch.Tensor:
        return (1-(1-sigma_min)*t)*x + t*x_1
    
    def compute_dx_t(self, x, x_1, sigma_min) -> torch.Tensor:
        return x_1 - (1 - sigma_min)*x
    
    def loss(self, model: torch.nn,  x_1, sigma_min = 0) -> torch.Tensor:
        
        x_0 = torch.randn_like(x_1, device=x_1.device, dtype=x_1.dtype)        
        t = torch.rand(len(x_1), 1, device=x_1.device, dtype=x_1.dtype)

        dx_t = self.compute_dx_t(x_0, x_1, sigma_min)
        x_t = self.compute_x_t(x_0, x_1, sigma_min, t)   
        
        v_pred = model(x_t, t)    
        
        loss = torch.mean((v_pred - dx_t)**2)
        
        return loss
    
