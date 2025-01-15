from torch import nn, Tensor
import torch


class SimpleNN(nn.Module):
    def __init__(self, dim:int = 2, h:int=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, h), 
            nn.ELU(),
            nn.Linear(h,h), 
            nn.ELU(),
            nn.Linear(h,h), 
            nn.ELU(),
            nn.Linear(h,dim)
        )
        
    def forward(self, x_t:Tensor, t:Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end:Tensor) -> Tensor:
        t_start = t_start.view(1,1).expand(x_t.shape[0], 1)
        
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end-t_start)/2,
                                              t_start + (t_end-t_start)/2) 