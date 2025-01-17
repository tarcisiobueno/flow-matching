
from matplotlib import pyplot as plt
import torch
from torch import nn
import numpy as np
from torchdiffeq import odeint
from models.models import ModelWrapper


class Sampler2D:
    def __init__(self, model: nn.Module, x_0, time_steps=10, method="dopri5", device=None):
        """
        Initializes the Sampler2D class and computes the solution.

        Parameters:
        - model (nn.Module): The model representing the dynamics.
        - x_0 (torch.Tensor): Initial state, shape [samples, dimensions].
        - device (torch.device or str): The device to run the model and computations on (default is None, which uses the CPU).
        - time_steps (int): Number of time steps to solve for.
        - method (str): ODE solver method (default is 'dopri5').
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelWrapper(model).to(self.device) 
        self.x_0 = x_0.to(self.device)
        self.time_steps = time_steps
        self.method = method

        # Compute the solution during initialization
        self.solution = self.compute_solution(self.time_steps)

    def compute_solution(self, time_steps = 2000):
        """
        Computes the solution of the dynamics using the given model.
        """
        t = torch.linspace(0, 1, time_steps, device=self.device)
        
        # Wrap the model to match the signature required by odeint
        
        return odeint(self.model, self.x_0, t, method=self.method)

    def plot_sample(self, title="Sample from target distribution", figsize=(8, 6), point_size=15, alpha=0.7, color='blue'):
        """
        Plots a 2D scatter plot for the final time step of the solution.

        Parameters:
        - title (str): Title of the plot.
        - figsize (tuple): Size of the figure (width, height).
        - point_size (float): Size of the scatter points.
        - alpha (float): Transparency of the scatter points.
        - color (str or list): Color of the scatter points.
        """
        # Extract the last time step and dimensions
        x = self.solution[-1, :, 0].detach().cpu().numpy()
        y = self.solution[-1, :, 1].detach().cpu().numpy()

        # Create the plot
        plt.figure(figsize=figsize)
        plt.scatter(x, y, s=point_size, alpha=alpha, color=color)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio for axes
        plt.tight_layout()
        plt.show()

    def plot_flow(self, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis'):
        """
        Plots a series of subplots showing the distribution of samples at each time step.

        Parameters:
        - figsize (tuple): Size of the overall figure (width, height).
        - point_size (float): Size of the scatter points.
        - alpha (float): Transparency of the scatter points.
        - cmap (str): Colormap for the scatter points.
        """
        # Extract all time steps and dimensions
        x_values = self.solution[:, :, 0].detach().cpu().numpy()
        y_values = self.solution[:, :, 1].detach().cpu().numpy()
        time_steps = len(self.solution)

        # Create subplots
        fig, axes = plt.subplots(1, time_steps, figsize=figsize, sharex=True, sharey=True)

        for i in range(time_steps):
            ax = axes[i]
            sc = ax.scatter(
                x_values[i], 
                y_values[i],
                s=point_size,
                alpha=alpha,
                c=[i / (time_steps - 1)] * len(x_values[i]),
                cmap=cmap
            )
            ax.set_title(f"t = {i / (time_steps - 1):.2f}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')    
        
        plt.tight_layout()
        plt.show()
        
    def plot_trajectories(self, n=2000):
        """Plot trajectories of some selected samples."""
        solution_traj = self.compute_solution(time_steps=2000)
        solution_traj_np = solution_traj.detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(solution_traj_np[0, :n, 0], solution_traj_np[0, :n, 1], s=4, alpha=0.8, c="black")
        plt.scatter(solution_traj_np[:, :n, 0], solution_traj_np[:, :n, 1], s=0.05, alpha=0.05, c="whitesmoke")
        plt.scatter(solution_traj_np[-1, :, 0], solution_traj_np[-1, :, 1], s=10, alpha=1, c="navy")
        plt.legend(["x0 ~ p", "Flow", "x1 ~ q"])
        plt.xticks([])
        plt.yticks([])
        plt.show()