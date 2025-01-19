import numpy as np
from torchdiffeq import odeint
from matplotlib import pyplot as plt
import torch
from torch import nn
from models.models import ModelWrapper
from matplotlib.lines import Line2D

from torchdiffeq import odeint_event

class Sampler2D:
    def __init__(self, model: nn.Module, x_0, time_steps=10, fm_method="default",ode_method="dopri5", device=None):
        """
        Initializes the Sampler2D class and computes the solution.

        Parameters:
        - model (nn.Module): The model representing the dynamics.
        - x_0 (torch.Tensor): Initial state, shape [samples, dimensions].
        - device (torch.device or str): The device to run the model and computations on (default is None, which uses the CPU).
        - time_steps (int): Number of time steps to solve for.
        - ode_method (str): ODE solver ode_method (default is 'dopri5').
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelWrapper(model, count_nfe=True).to(self.device) 
        self.x_0 = x_0.to(self.device)
        self.time_steps = time_steps
        self.ode_method = ode_method
        self.fm_method = fm_method

        # Compute the solution during initialization
        self.solution = self.compute_solution(self.time_steps)

    def compute_solution(self, time_steps = 10):
        """
        Computes the solution of the dynamics using the given model.
        """
        if self.fm_method == "diffusion":
            t = torch.linspace(0, 1-1e-5, time_steps, device=self.device)
        else:
            t = torch.linspace(0, 1, time_steps, device=self.device)
        with torch.no_grad():
            solution = odeint(self.model, self.x_0, t, method=self.ode_method)
        return solution
    
    def compute_solution_from_t_to_t1(self, t, t1):
        """
        Computes the solution of the dynamics using the given model from t to t1.
        """
        assert t < t1, "t must be less than t1"
        if self.fm_method == "diffusion":
            assert t1 <= 1-1e-5, "t1 must be less than or equal to 1-1e-5"
        else:
            assert t1 <= 1, "t1 must be less than or equal to 1"
        assert t >= 0, "t must be greater than or equal to 0"
        
        t = torch.Tensor([t,t1]).to(self.device)
        with torch.no_grad():
            # atol=1e-5, rtol=1e-5 as in Lipman et al. 2023
            solution = odeint(self.model, self.x_0, t, method=self.ode_method, rtol=1e-5,  atol=1e-5)
        return solution

    def plot_sample(self, title="Sample from target distribution", figsize=(8, 6), point_size=15, alpha=0.7, color='#470756'):
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


    def plot_trajectories(self, n=2000, title=""):
        """Plot trajectories of some selected samples."""
        solution_traj = self.compute_solution(time_steps=5000)
        solution_traj_np = solution_traj.detach().cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.scatter(solution_traj_np[0, :n, 0], solution_traj_np[0, :n, 1], s=4, alpha=0.8, c="black")
        plt.plot(solution_traj_np[:, :n, 0], solution_traj_np[:, :n, 1], linestyle="--", color="black", alpha=1, linewidth=0.3)
        plt.scatter(solution_traj_np[-1, :, 0], solution_traj_np[-1, :, 1], s=10, alpha=1, c="navy")
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='x_0 ~ p'),
            Line2D([0], [0], color='black', linestyle="--", linewidth=0.3, label='Path'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=10, label='x_1 ~ q')
        ]
        plt.title(title)
        plt.legend(handles=legend_handles)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        

    def compute_solution_nfe_target(self, nfe_target=4, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis'):
        """
        Computes the solution of the dynamics using the given model until the NFE target is reached.
        """
        self.model.reset_nfe()

        # Define an event function based on NFE
        def event_fn(t, y):
            # Stop when the NFE counter reaches the limit
            return torch.tensor([self.model.nfe - nfe_target], device=self.device)
        
        t = torch.Tensor(1).to(self.device)
        # Solve ODE with event detection
        with torch.no_grad():
            solution = odeint_event(
                self.model,
                self.x_0,
                t,
                event_fn=event_fn,
                reverse_time=False,
                odeint_interface=odeint,
                method=self.ode_method, 
                rtol=1e-5,
                atol=1e-5
            )

        return solution

    def compute_solution_time_target(self, time_target=0.5, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis'):
        """
        Computes the solution of the dynamics using the given model until the NFE target is reached.
        """
        def event_fn(t, y):
            # Stop when time reaches the target
            return torch.tensor([time_target-t], device=self.device)
        
        t = torch.Tensor(1).to(self.device)
        # Solve ODE with event detection
        with torch.no_grad():
            solution = odeint_event(
                self.model,
                self.x_0,
                t,
                event_fn=event_fn,
                reverse_time=False,
                odeint_interface=odeint,
                method=self.ode_method, 
                rtol=1e-5,
                atol=1e-5
            )

        return solution
    
    def plot_flow_nfe_targets(self, nfe_targets, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis', title="", title_y=0.75):
        """
        Plots a series of subplots showing the distribution of samples at each NFE target.

        Parameters:
        - nfe_targets (list): List of NFE targets to reach.
        - figsize (tuple): Size of the overall figure (width, height).
        - point_size (float): Size of the scatter points.
        - alpha (float): Transparency of the scatter points.
        - cmap (str): Colormap for the scatter points.
        """
        solutions = []
        times = []
        n_nfe_targets = len(nfe_targets)
        
        for nfe_target in nfe_targets:
            solution = self.compute_solution_nfe_target(nfe_target)
            solutions.append(solution[1][-1])
            times.append(solution[0])

        solutions_array = torch.stack(solutions).detach().cpu().numpy()
        times_array = torch.stack(times).detach().cpu().numpy()
    

        x_values = solutions_array[:, :, 0]
        y_values = solutions_array[:, :, 1]

        # Create subplots
        fig, axes = plt.subplots(1, n_nfe_targets, figsize=figsize, sharex=True, sharey=True)
        fig.suptitle(title, y=title_y, fontsize=16)
        # If there is only one subplot, axes will be a single Axes object
        if n_nfe_targets == 1:
            axes = [axes]

        for i in range(n_nfe_targets):
            ax = axes[i]
            sc = ax.scatter(
                x_values[i], 
                y_values[i],
                s=point_size,
                alpha=alpha,
                c=[i / (n_nfe_targets - (1-1e-5))] * len(x_values[i]),
                cmap=cmap
            )
            ax.set_title(f"NFE = {nfe_targets[i]}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')    

        plt.tight_layout()
        plt.show()

    def plot_flow_time_targets(self, time_targets, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis', title="", title_y=0.75):
        """
        Plots a series of subplots showing the distribution of samples at each NFE target.

        Parameters:
        - nfe_targets (list): List of NFE targets to reach.
        - figsize (tuple): Size of the overall figure (width, height).
        - point_size (float): Size of the scatter points.
        - alpha (float): Transparency of the scatter points.
        - cmap (str): Colormap for the scatter points.
        """
        solutions = []
        times = []
        n_time_targets = len(time_targets)
        
        for t_target in time_targets:
            solution = self.compute_solution_time_target(t_target)
            solutions.append(solution[1][-1])
            times.append(solution[0])

        solutions_array = torch.stack(solutions).detach().cpu().numpy()
        times_array = torch.stack(times).detach().cpu().numpy()

        x_values = solutions_array[:, :, 0]
        y_values = solutions_array[:, :, 1]

        # Create subplots
        fig, axes = plt.subplots(1, n_time_targets, figsize=figsize, sharex=True, sharey=True)
        fig.suptitle(title, y=title_y, fontsize=16)
        # If there is only one subplot, axes will be a single Axes object
        if n_time_targets == 1:
            axes = [axes]

        for i in range(n_time_targets):
            ax = axes[i]
            sc = ax.scatter(
                x_values[i], 
                y_values[i],
                s=point_size,
                alpha=alpha,
                c=[i / (n_time_targets - (1-1e-5))] * len(x_values[i]),
                cmap=cmap
            )
            ax.set_title(f"t = {times_array[i]:.2f}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')    

        plt.tight_layout()
        plt.show()
    
    def plot_flow(self, time_steps, plot_nfe=True, figsize=(16, 8), point_size=10, alpha=0.7, cmap='viridis', title=""):
        """
        Plots a series of subplots showing the distribution of samples at each time step.

        Parameters:
        - figsize (tuple): Size of the overall figure (width, height).
        - point_size (float): Size of the scatter points.
        - alpha (float): Transparency of the scatter points.
        - cmap (str): Colormap for the scatter points.
        """
        nfe_results = [] 
        solutions = []
        if self.fm_method == "diffusion":
            nfe_times = torch.linspace(0, 1-1e-5, time_steps, device=self.device)
        else:
            nfe_times = torch.linspace(0, 1, time_steps, device=self.device)
        
        for i in nfe_times[1:]:
            self.model.nfe = 0
            solutions.append(self.compute_solution_from_t_to_t1(0, i)[-1])
            nfe_results.append(self.model.nfe)

        nfe_results_array = np.array(nfe_results)
        solutions_array = torch.stack(solutions).detach().cpu().numpy()
        
        x_values = solutions_array[:, :, 0]
        y_values = solutions_array[:, :, 1]
        time_steps = len(nfe_results_array)

        # Create subplots
        fig, axes = plt.subplots(1, time_steps, figsize=figsize, sharex=True, sharey=True)
        fig.suptitle(title, y=0.7, fontsize=16)

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
            if plot_nfe:
                ax.set_title(f"t = {(nfe_times[i+1]):.2f} - NFE: {nfe_results_array[i]}")
            else:
                ax.set_title(f"t = {(nfe_times[i+1]):.2f}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_aspect('equal', adjustable='box')    
        
        plt.tight_layout()
        plt.show()