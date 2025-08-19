import numpy as np
from sklearn.datasets import make_moons
from .base_dataset_class import BaseDataset2D

class MoonWithScatteringsDataset(BaseDataset2D):
    noise_level=0.1
    crescent_class=0
    scatter_band=0.3
    scatter_density=500
    
    def __init__(self, num_samples=10000, dataset_name='Moon_with_scatterings', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)
        
    def pdf(self, distribution):
        if distribution == "standard_normal":
            return self.random_standard_normal()
    
    def generate_crescent_with_scatter(self):
        X, y = make_moons(2*self.num_samples - 2*self.scatter_density, noise=self.noise_level, random_state=self.random_state)
        X_single_crescent = X[y == self.crescent_class]
        
        X_scatter = []
        for point in X_single_crescent:
            x_crescent, y_crescent = point

            # Add noise to the crescent points within a specified scatter band
            x_scattered = x_crescent + self.scatter_band * self.pdf(distribution="standard_normal")
            y_scattered = y_crescent + self.scatter_band * self.pdf(distribution="standard_normal")
            X_scatter.append((x_scattered, y_scattered))
        
        
        X_scatter = np.array(X_scatter[:self.scatter_density])  # Limit the scatter points to the specified density
        X_combined = np.vstack((X_single_crescent, X_scatter))
        return X_combined
    
    def generate(self):
        X = self.generate_crescent_with_scatter()
        X = self.normalize(X)
        self.data = X
        return X