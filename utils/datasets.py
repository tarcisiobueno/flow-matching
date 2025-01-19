import numpy as np
from sklearn.datasets import make_swiss_roll, make_moons
from sklearn.discriminant_analysis import StandardScaler
import torch

def create_dataloader(data, batch_size=128):
    """
    Creates a PyTorch DataLoader from the provided data.

    Parameters:
    - data (numpy.ndarray): Input data to be converted to a tensor.
    - batch_size (int, optional): Batch size for the DataLoader. Default is 128.
    - device (str, optional): Device to move the data to ('cpu' or 'cuda'). Default is 'cuda'.

    Returns:
    - dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
    """
    # Convert the data to a PyTorch tensor
    dataset = torch.from_numpy(data).float()
    
    # Create a TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return dataloader


def get_data(dataset: str, n_points: int) -> np.ndarray:
    """
    Generate and return a standardized dataset.

    Parameters:
    dataset (str): The type of dataset to generate. Options are "moons", "swiss", or "checkers".
    n_points (int): The number of data points to generate.

    Returns:
    np.ndarray: The standardized dataset.

    Raises:
    ValueError: If the dataset type is not recognized.
    """
    if dataset == "moons":
        data, _ = make_moons(n_points)
    elif dataset == "swiss":
        data, _ = make_swiss_roll(n_points)
        data = data[:, [0, 2]] / 10.0
        
    # adapted from https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb
    elif dataset == "checkers":
        x1 = torch.rand(n_points) * 4 - 2
        x2_ = torch.rand(n_points) - torch.randint(high=2, size=(n_points,)) * 2
        x2 = x2_ + (torch.floor(x1) % 2)
        
        data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
        data = data.numpy()
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    return StandardScaler().fit_transform(data)