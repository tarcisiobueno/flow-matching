
from matplotlib import pyplot as plt
import torch
import numpy as np
from torchdiffeq import odeint



def sample_flow_trough_time_2d_old(flow, n_samples=256, n_steps=10, solver='dopri5'):
    
    # write version with dopri5
    
    
    
    # default solver
    # Generate initial samples
    x = torch.randn(n_samples, 2)
    
    # Create subplots
    fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
    time_steps = torch.linspace(0, 1.0, n_steps + 1)
    
    # Plot initial samples
    axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[0].set_title(f't={time_steps[0]:.2f}')
    axes[0].set_xlim(-3.0, 3.0)
    axes[0].set_ylim(-3.0, 3.0)
    
    # Perform steps and plot samples at each step
    for i in range(n_steps):
        x = flow.step(x, time_steps[i], time_steps[i + 1])
        axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[i + 1].set_title(f't={time_steps[i + 1]:.2f}')
    
    plt.tight_layout()
    plt.show()


# Wrapper class for the ODE function
class ODEFunc(torch.nn.Module):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow
        
    def forward(self, t, x):
        # Reshape t for the flow model
        t = torch.ones(x.shape[0]) * t
        return self.flow(x, t)

def sample_flow_trough_time_2d(flow, n_samples=256, n_steps=10, solver='dopri5'):
    
    x = torch.randn(n_samples, 2)
    time_steps = torch.linspace(0, 1.0, n_steps + 1)
    
    if solver == 'dopri5':        
        # Create ODE function
        ode_func = ODEFunc(flow)
        
        # Solve ODE using dopri5
        with torch.no_grad():
            trajectory = odeint(
                ode_func,
                x,
                time_steps,
                method='dopri5',
                rtol=1e-3,
                atol=1e-3
            )
        
        # Create subplots
        fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
        
        # Plot samples at each time step
        for i in range(n_steps + 1):
            axes[i].scatter(
                trajectory[i].detach()[:, 0],
                trajectory[i].detach()[:, 1],
                s=10
            )
            axes[i].set_title(f't={time_steps[i]:.2f}')
            axes[i].set_xlim(-3.0, 3.0)
            axes[i].set_ylim(-3.0, 3.0)
            
    else:
        # Default solver using step function
        # Create subplots
        fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
        
        # Plot initial samples
        axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
        axes[0].set_title(f't={time_steps[0]:.2f}')
        axes[0].set_xlim(-3.0, 3.0)
        axes[0].set_ylim(-3.0, 3.0)
        
        # Perform steps and plot samples at each step
        for i in range(n_steps):
            x = flow.step(x, time_steps[i], time_steps[i + 1])
            axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
            axes[i + 1].set_title(f't={time_steps[i + 1]:.2f}')
    
    plt.tight_layout()
    plt.show()


