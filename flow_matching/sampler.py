from torchdiffeq import odeint
import torch, torchvision
import numpy as np

class FlowMatchingSampler:
    def __init__(self, net_model, device='cuda'):
        self.net_model = net_model
        self.device = device
        
    def vector_field(self, t, x, num_channels=3):
        """Vector field for ODE solver"""
        # Reshape t to match the expected input shape of the model
        t_shaped = t.reshape(1, 1, 1, 1).expand(x.shape[0], num_channels, x.shape[2], x.shape[3]).to(self.device)
        return self.net_model(t_shaped, x)
    
    def generate_samples(self, num_samples, rtol=1e-5, atol=1e-5):
        """
        Generate samples using the trained flow matching model
        
        Args:
            num_samples: Number of samples to generate
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
        """
        # Initial noise
        x_0 = torch.randn(num_samples, 3, 32, 32).to(self.device)
        
        # Time points for integration
        t = torch.linspace(0, 1, 2).to(self.device)
        
        # Solve ODE
        self.net_model.eval()
        with torch.no_grad():
            samples = odeint(
                self.vector_field,
                x_0,
                t,
                method='dopri5',
                rtol=rtol,
                atol=atol
            )
        
        # Return final samples (at t=1)
        return samples[-1]

def save_samples(samples, filename):
    """Helper function to save generated samples"""
    # Ensure samples are in correct range [0, 1]
    samples = torch.clamp(samples, 0, 1)
    # Save as grid of images
    torchvision.utils.save_image(
        samples,
        filename,
        nrow=int(np.sqrt(len(samples))),
        normalize=True
    )

