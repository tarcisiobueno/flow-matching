from torch import nn, Tensor
import torch
from torch.nn import functional as F

class ModelWrapper(nn.Module):
    '''
    A wrapper for neural network models to add functionality for counting 
    the number of function evaluations (NFEs) and adapting time inputs.
    
    Attributes:
    ----------
    model : nn.Module
        The neural network model to be wrapped.
    nfe : int
        Counter for the number of function evaluations (forward passes).
    count_nfe : bool
        Flag to enable or disable counting NFEs.

    Methods:
    -------
    reset_nfe():
        Resets the NFE counter to zero.

    forward(t, x, *args, **kwargs):
        Performs a forward pass through the wrapped model
    '''
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


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + 1, base_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ELU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.ELU()
        )
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 6, base_channels * 2, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Expand time tensor to match spatial dimensions
        b, c, h, w = x_t.shape
        t_expanded = t.expand(-1, 1, h, w)

        # Concatenate time as an additional channel
        x = torch.cat((x_t, t_expanded), dim=1)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc2, 2))

        # Decoder with skip connections
        dec2 = self.dec2(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False), enc1], dim=1))

        return dec1

    def step(self, x_t: torch.Tensor, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        dt = t_end - t_start
        v_mid = self(x_t + self(x_t, t_start) * dt / 2, t_start + dt / 2)
        return x_t + dt * v_mid