import numpy as np
from sklearn.datasets import make_moons
from .base_dataset_class import BaseDataset2D

class MoonWithTwoCirclesUnboundedDataset(BaseDataset2D):
    noise_level=0.1
    crescent_class=0 
    circle_center1=(-1.5, -1.0)
    circle_center2=(1.5, -1.0)
    circle_radius=0.2
    circle_density=500
   
    def __init__(self, num_samples=10000, dataset_name='Moon_with_circles_unbounded', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)
        
    def pdf(self, distribution, params, size=None):
        if distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
        elif distribution == "standard_normal":
            return self.random_standard_normal(size)

    def generate_crescent_with_circles(self):
        X, y = make_moons(2*self.num_samples-4*self.circle_density, noise=self.noise_level, random_state=self.random_state)
        X_single_crescent = X[y == self.crescent_class]
        
        angles1 = 2 * np.pi * self.pdf(distribution="uniform", size=self.circle_density, params={'low':0, 'high':1})  # Random angles for circular distribution
        angles2 = 2 * np.pi * self.pdf(distribution="uniform", size=self.circle_density, params={'low':0, 'high':1})
        radii1 = self.circle_radius * np.sqrt(self.pdf(distribution="uniform", size=self.circle_density, params={'low':0, 'high':1}))  # Scaled radii for dense clustering
        radii2 = self.circle_radius * np.sqrt(self.pdf(distribution="uniform", size=self.circle_density, params={'low':0, 'high':1}))
        
        radii1 += 0.05 * self.pdf(distribution="standard_normal", size=self.circle_density, params = {})
        radii2 += 0.05 * self.pdf(distribution="standard_normal", size=self.circle_density, params = {})
        # Convert polar coordinates to Cartesian for the small circle points
        X_circle1 = np.c_[radii1 * np.cos(angles1) + self.circle_center1[0], radii1 * np.sin(angles1) + self.circle_center1[1]]
        X_circle2 = np.c_[radii2 * np.cos(angles2) + self.circle_center2[0], radii2 * np.sin(angles2) + self.circle_center2[1]]

        # Combine crescent and small circular cluster points
        X_combined = np.vstack((X_single_crescent, X_circle1, X_circle2))
        
        return X_combined

    def generate(self):
        X = self.generate_crescent_with_circles()
        X = self.normalize(X)
        self.data = X
        return X