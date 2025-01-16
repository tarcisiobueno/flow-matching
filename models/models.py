from torch import nn, Tensor
import torch

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):

        t_reshaped = torch.reshape(t.expand(x.shape[0]),(-1,1))       

        return self.model(x, t_reshaped)

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )
    

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)
    
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