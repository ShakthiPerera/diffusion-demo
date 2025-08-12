import numpy as np
from .base_dataset_class import BaseDataset2D

class CentralBananaDataset(BaseDataset2D):
    central_point = 500
    
    def __init__(self, num_samples=10000, dataset_name='Central_Banana', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)

def banana_cen_pdf(self, theta, max_distance, concentration_factor=2, decay=0.5):
    r = max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))
    return r

def pdf(self, distribution, params, size=None):
    """
    Generates random numbers based on the specified distribution and parameters.

    :param distribution: str, type of distribution ('exponential', 'uniform', 'standard_normal')
    :param params: dict, parameters required for the distribution
    :param size: int or tuple, shape of the output array
    :return: Generated random numbers
    """
    
    if distribution == "exponential":
        return self.random_exponential(params["scale"], size)
    elif distribution == "uniform":
        return self.random_uniform(params["low"], params["high"], size)
    elif distribution == "standard_normal":
        return self.random_standard_normal(size)
    elif distribution == "normal":
        return self.random_normal(params["mean"], params["std"])

def generate_cen_banana_points(self, num_points, sigma=0.3, max_distance=1.0):
    X = []
    for _ in range(num_points):
        theta = self.pdf(distribution="exponential", params={"scale":0.25})
        r = banana_cen_pdf(theta, max_distance)
        width_effect = sigma * (1 - (r / max_distance))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})
        y_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})
        x_new = x_center + x_offset
        y_new = y_center + y_offset
        X.append((x_new, y_new))
    return np.array(X)

def generate_central_points(self, num_points, max_distance=0.5, sigma=0.1):
    central_points = []
    for _ in range(num_points):
        theta = np.pi / 3 + self.pdf(distribution="normal", params={"mean":0, "std":0.5})
        r = max_distance * (self.pdf(distribution="uniform", params={"low":0.5, "high":0.8}))
        x_center = r * np.cos(theta)
        y_center = r * np.sin(theta)
        x_offset = self.pdf(distribution="normal", params={"mean":0, "std":sigma / 2})
        y_offset = self.pdf(distribution="normal", params={"mean":0, "std":sigma / 2})
        central_points.append((x_center + x_offset, y_center + y_offset))
    return np.array(central_points)

def generate(self):
    banana_points = generate_cen_banana_points(num_points=self.num_samples - self.central_points)
    middle_points = generate_central_points(num_points=self.central_points)
    all_points = np.vstack((banana_points, middle_points))
    X = self.normalize(all_points)
    self.data = X
    return X