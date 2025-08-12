import numpy as np
from .base_dataset_class import BaseDataset2D

class BananaWithTwoCirclesDataset(BaseDataset2D):
    circle_center1=(-0.4, 0.5)
    circle_center2=(0.5, -0.25)
    circle_radius=0.075
    circle_density=500
    circle_noise=0.05
    scatter_density=500
    
    def __init__(self, num_samples=10000, dataset_name='Banana_with_two_circles', random_state=42, noise_level=0.1):
        super().__init__(num_samples, dataset_name, random_state)
        self.noise_level = noise_level
        
    def banana_pdf(self, theta, max_distance, concentration_factor=2, decay=0.5):
        return max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))

    def pdf(self, distribution, params, size=None):
        if distribution == "exponential":
            return self.random_exponential(params["scale"], size)
        elif distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
        elif distribution == "standard_normal":
            return self.random_standard_normal(size)
    
    def generate_banana_points(self, sigma=0.3, max_distance=1.0):
        X = []
        for _ in range(self.num_samples):
            theta = self.pdf(distribution="exponential", params={"scale":0.2})
            r = self.banana_pdf(theta, max_distance)
            width_effect = sigma * (1 - (r / max_distance))

            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)

            x_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})
            y_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})
            X.append((x_center + x_offset, y_center + y_offset))

        return np.array(X)

    def generate_banana_with_noisy_circles_and_local_scatter(self):
        X_banana = self.generate_banana_points(self.num_samples - 2 * self.circle_density - self.scatter_density)

        angles1 = 2 * np.pi * self.pdf(distribution="uniform", size=self.circle_density)
        angles2 = 2 * np.pi * self.pdf(distribution="uniform", size=self.circle_density)
        radii1 = self.circle_radius * np.sqrt(self.pdf(distribution="uniform", size=self.circle_density)) + self.circle_noise * self.pdf(distribution="standard_normal", size=self.circle_density)
        radii2 = self.circle_radius * np.sqrt(self.pdf(distribution="uniform", size=self.circle_density)) + self.circle_noise * self.pdf(distribution="standard_normal",size=self.circle_density)
        
        X_circle1 = np.c_[radii1 * np.cos(angles1) + self.circle_center1[0], radii1 * np.sin(angles1) + self.circle_center1[1]]
        X_circle2 = np.c_[radii2 * np.cos(angles2) + self.circle_center2[0], radii2 * np.sin(angles2) + self.circle_center2[1]]

        X_scatter = []
        for _ in range(self.scatter_density):
            theta = self.pdf(distribution="exponential", params={"scale":0.2})
            r = self.banana_pdf(theta, max_distance=1.0) * (1 + self.noise_level * self.pdf(distribution="standard_normal"))

            x_noisy = r * np.cos(theta)
            y_noisy = r * np.sin(theta)
            X_scatter.append((x_noisy, y_noisy))

        X_scatter = np.array(X_scatter)

        X_combined = np.vstack((X_banana, X_circle1, X_circle2, X_scatter))

        return X_combined
    
    def generate(self):
        X = self.generate_banana_with_noisy_circles_and_local_scatter()
        X = self.normalize(X)
        self.data = X
        return X