from torch import nn, Tensor
import torch

class ModelWrapper(nn.Module):
    def __init__(self, model, count_nfe=False):
        super().__init__()
        self.model = model
        self.nfe = 0  # Number of function evaluations
        self.count_nfe = count_nfe
        
    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        if self.count_nfe:
            self.nfe += 1
            
        t_reshaped = torch.reshape(t.expand(x.shape[0]),(-1,1))     

        return self.model(x, t_reshaped)

class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, output_dim: int = 2, h:int=64, num_layers:int =5):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim+1, h))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers-2):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(h, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x_t:Tensor, t:Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end:Tensor) -> Tensor:
        t_start = t_start.view(1,1).expand(x_t.shape[0], 1)
        
        return x_t + (t_end - t_start) * self(x_t + self(x_t, t_start) * (t_end-t_start)/2,
                                              t_start + (t_end-t_start)/2) 
    