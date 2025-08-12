import numpy as np
from .base_dataset_class import BaseDataset2D

class TwoRingsBoundedDataset(BaseDataset2D):
    def __init__(self, num_samples=10000, dataset_name='Two_rings_bounded', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)
    
    def pdf(self, distribution, params, size=None):
        if distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
    
    def generate_ring_points(self, center, inner_radius, outer_radius, num_points):
        angles = 2 * np.pi * self.pdf(distribution="uniform", params={"low": 0, "high": 1}, size=num_points)  # Random angles
        radii = np.sqrt(self.pdf(distribution="uniform", params={"low": inner_radius**2, "high": outer_radius**2}, size=num_points))  # Random radii between inner and outer circles
        X_ring = np.c_[radii * np.cos(angles) + center[0], radii * np.sin(angles) + center[1]]  # Cartesian coordinates
        return X_ring

    def create_two_ring_dataset(self, center=(0, 0), inner_circle_radius1=0.1, outer_circle_radius1=0.2, inner_circle_radius2=0.3, outer_circle_radius2=0.4):
        # Generate points for the first ring (using inner and outer radii)
        X_ring1 = self.generate_ring_points(center, inner_circle_radius1, outer_circle_radius1, self.num_samples//2)

        # Generate points for the second ring (with the same center and radii)
        X_ring2 = self.generate_ring_points(center, inner_circle_radius2, outer_circle_radius2, self.num_samples//2)

        # Combine the two sets of points to form two distinct rings
        X_combined = np.vstack((X_ring1, X_ring2))
        
        return X_combined
    
    def generate(self):
        X = self.create_two_ring_dataset()
        X = self.normalize(X)
        self.data = X
        return X
