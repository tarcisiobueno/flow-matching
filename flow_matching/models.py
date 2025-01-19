from matplotlib import pyplot as plt
import torch

# More details about the algorithm can be found in the papers:
# https://arxiv.org/abs/2106.04946 and https://arxiv.org/abs/2412.06264
    
class DiffusionFM:
    '''
    Implements a Variance-Preserving Diffusion Flow Matching Matching.   
     
    Attributes:
    ----------
    beta_min : float
        The minimum beta value for the linear beta schedule.
    beta_max : float
        The maximum beta value for the linear beta schedule.
    eps : float
        A small value to prevent numerical issues (default is 1e-5).
    minist : bool
        True if the data is MNIST, False otherwise.
    '''
    
    def __init__(self, beta_min=0.1, beta_max=20, minist=False):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = 1e-5
        self.minist = minist

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
        if self.minist:
            t = t.view(-1, 1, 1, 1)
        x_t = self.compute_x_t(x_0, x_1, t)
        dx_t = self.compute_dx_t(x_t, x_1, t)
        v_pred = model(x_t, t)      
          
        return torch.mean((v_pred - dx_t)**2)

# Flow matching with optimal transport
# COTFM - Conditional Optimal Transport Flow Matching
class COTFM:
    '''
    Class implementing Conditional Optimal Transport Flow Matching (COTFM).
        
    Attributes:
    ----------
    minist : bool
        True if the data is MNIST, False otherwise.

    Methods:
    -------
    compute_x_t(x: torch.Tensor, x_1: torch.Tensor, sigma_min, t) -> torch.Tensor:
        Computes the interpolated state x_t between x and x_1 at time t.

    compute_dx_t(x: torch.Tensor, x_1: torch.Tensor, sigma_min) -> torch.Tensor:
        Computes the velocity field (dx_t) based on the start and target states (x and x_1).

    loss(model: torch.nn.Module, x_1: torch.Tensor, sigma_min=0) -> torch.Tensor:
        Computes the training loss for the given model based on flow matching principles.
    '''
    def __init__(self, minist=False):
        self.minist = minist 

    def compute_x_t(self, x:torch.Tensor, x_1:torch.Tensor, sigma_min, t) -> torch.Tensor:
        return (1-(1-sigma_min)*t)*x + t*x_1
    
    def compute_dx_t(self, x, x_1, sigma_min) -> torch.Tensor:
        return x_1 - (1 - sigma_min)*x
    
    def loss(self, model: torch.nn,  x_1, sigma_min = 0) -> torch.Tensor:
        
        x_0 = torch.randn_like(x_1, device=x_1.device, dtype=x_1.dtype)        
        t = torch.rand(len(x_1), 1, device=x_1.device, dtype=x_1.dtype)
        
        if self.minist:
            t = t.view(-1, 1, 1, 1)
        dx_t = self.compute_dx_t(x_0, x_1, sigma_min)
        x_t = self.compute_x_t(x_0, x_1, sigma_min, t)   
        
        v_pred = model(x_t, t)    
        
        loss = torch.mean((v_pred - dx_t)**2)
        
        return loss
    

def visualize_mnist_flow(model, n_samples=10, n_steps=8, device='cuda'):
    """
    Visualize the flow from noise to MNIST-like images.

    Args:
        model: The trained model
        n_samples: Number of images to generate
        n_steps: Number of intermediate steps to show
        device: Device to run the model on
    """

    model.eval()

    # Generate initial random noise
    x = torch.randn(n_samples, 1, 28, 28).to(device)

    # Create subplots
    fig, axes = plt.subplots(n_samples, n_steps + 1, figsize=(2*(n_steps + 1), 2*n_samples))
    time_steps = torch.linspace(0, 1.0, n_steps + 1)

    # Plot initial noise samples
    for j in range(n_samples):
        axes[j, 0].imshow(x[j, 0].cpu().detach(), cmap='gray')
        axes[j, 0].axis('off')
    axes[0, 0].set_title(f't={time_steps[0]:.2f}')

    # Perform steps and plot samples at each step
    with torch.no_grad():
        for i in range(n_steps):
            t_start = torch.tensor([time_steps[i]], device=device).view(1, 1, 1, 1).expand(n_samples, -1, -1, -1)
            t_end = torch.tensor([time_steps[i + 1]], device=device).view(1, 1, 1, 1).expand(n_samples, -1, -1, -1)

            x = model.step(x, t_start, t_end)

            # Plot each sample
            for j in range(n_samples):
                axes[j, i + 1].imshow(x[j, 0].cpu().detach(), cmap='gray')
                axes[j, i + 1].axis('off')
            axes[0, i + 1].set_title(f't={time_steps[i + 1]:.2f}')

    plt.tight_layout()
    plt.show()
    
