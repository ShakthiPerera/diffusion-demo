import numpy as np
from .base_dataset_class import BaseDataset2D

class BananaDataset(BaseDataset2D):
    sigma = 0.3
    max_distance = 1.0
    
    def __init__(self, num_samples=10000, dataset_name='Banana', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)

    def banana_pdf(self, theta, concentration_factor=2, decay=0.5):
        """
        Define the radius for banana shape with density concentrated on one end.
        """
        # Make radius dependent on angle to create an elongated "banana" effect
        r = self.max_distance * (np.exp(-decay * theta) * (1 + concentration_factor * np.sin(theta)))
        return r

    def pdf(self, distribution, params, size=None):
        if distribution == "exponential":
            return self.random_exponential(params["scale"], size)
        elif distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
    
    def generate_banana_points(self):
        """
        Generate points according to the defined probability distribution for a banana shape.
        """
        X = []
        for _ in range(self.num_samples):
            theta = theta = self.pdf(distribution="exponential", params={"scale":0.25}) #np.random.uniform(0, np.pi/2)  # Random angle for half-circle (banana shape)
            r = self.banana_pdf(theta, self.max_distance)

            # Calculate width effect based on distance
            width_effect = self.sigma * (1 - (r / self.max_distance))  # Linearly decrease width with distance

            # Convert to Cartesian coordinates
            x_center = r * np.cos(theta)
            y_center = r * np.sin(theta)

            # Generate random offsets based on the current width
            x_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})
            y_offset = self.pdf(distribution="uniform", params={"low":-width_effect / 2, "high":width_effect / 2})

            # Add offsets to the center point
            x_new = x_center + x_offset
            y_new = y_center + y_offset

            # Add to the lists
            X.append((x_new, y_new))

        return np.array(X)

    def generate(self):
        X = self.generate_banana_points()
        X = self.normalized(X)
        self.data = X
        return X