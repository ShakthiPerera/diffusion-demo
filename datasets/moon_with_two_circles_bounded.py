import numpy as np
from sklearn.datasets import make_moons
from .base_dataset_class import BaseDataset2D

class MoonWithTwoCiclesBoundedDataset(BaseDataset2D):
    noise_level = 0.1
    crescent_class = 0
    crescent_bounds = (-2, 2, -1.5, 1.5)
    circle_center1 = (-1.5, -1.0)
    circle_center2 = (1.5, -1.0)
    circle_inner_radius = 0.01 
    circle_outer_radius = 0.25
    circle_density = 500
    
    def __init__(self, num_samples=10000, dataset_name='Moon_with_two_circles_bounded', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)
        
    def pdf(self, distribution, params, size=None):
        if distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
        elif distribution == "shuffle":
            return self.random_shuffle(params["data"])
        
    def generate_circle_points(self, center, inner_radius, outer_radius, num_points):
        angles = 2 * np.pi * self.pdf(distribution="uniform", size=num_points, params={"low": 0, "high":1})
        radii = np.sqrt(self.pdf(distribution="uniform", params={"low":inner_radius**2, "high":outer_radius**2}, size=num_points))
        X_circle = np.c_[radii * np.cos(angles) + center[0], radii * np.sin(angles) + center[1]]
        return X_circle 
        
    def generate_crescent_with_circles(self):
        # Generate 20% more data initially
        initial_samples = int(self.num_samples * 1.2)
        X, y = make_moons(n_samples=2 * initial_samples, noise=self.noise_level, random_state=self.random_state)
        X_single_crescent = X[y == self.crescent_class]
        
        X_circle1 = self.generate_circle_points(self.circle_center1, self.circle_inner_radius, self.circle_outer_radius, self.circle_density)
        X_circle2 = self.generate_circle_points(self.circle_center2, self.circle_inner_radius, self.circle_outer_radius, self.circle_density)

        # Combine crescent and circular cluster points
        X_combined = np.vstack((X_single_crescent, X_circle1, X_circle2))

        # Apply a single mask for bounding conditions
        mask = (
            (X_combined[:, 0] >= self.crescent_bounds[0]) & (X_combined[:, 0] <= self.crescent_bounds[1]) &
            (X_combined[:, 1] >= self.crescent_bounds[2]) & (X_combined[:, 1] <= self.crescent_bounds[3])
        )
        X_filtered = X_combined[mask]

        # Shuffle and select exactly `num_samples` points
        self.pdf(distribution="shuffle", params={"data":X_filtered})
        X_final = X_filtered[:self.num_samples]
        return X_final
    
    def generate(self):
        X = self.generate_crescent_with_circles()
        X = self.normalize(X)
        self.data = X
        return X