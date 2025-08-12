import numpy as np
from .base_dataset_class import BaseDataset2D

class StarFishUniformDataset(BaseDataset2D):
    def __init__(self, num_samples=10000, dataset_name='Star_fish_uniform', random_state=42):
        super().__init__(num_samples, dataset_name, random_state)

    def pdf(self, distribution, params, size=None):
        if distribution == "uniform":
            return self.random_uniform(params["low"], params["high"], size)
    
    def swirling_uni_pdf(self, theta, arm_index, max_distance, a=0.6, b=0.1, k=1, M=6):
        """
        Define the radius for swirling arms.
        """
        angle_offset = (2 * np.pi) / M  # Angle separation between each arm
        r_swirl = max_distance * (a * theta * (1 + b * np.sin(k * theta + arm_index * angle_offset)))
        return r_swirl

    def generate_uni_star_points(self, sigma=0.5, arm_size=5):
        """
        Generate points according to the defined probability distribution.
        """
        X = []

        # Calculate number of points for arms
        num_arm_points = self.num_samples // arm_size  # Points per arm
        remainder = self.num_samples % arm_size  # Calculate remainder points

        # Maximum distance for arms (to ensure points stay within -1 and 1)
        max_distance = 2.0  # Set max distance to 1 to constrain within bounds

        # Generate points for each arm
        for arm_index in range(arm_size):  # Assuming 5 arms
            # Distribute points evenly, and assign remainder to the first few arms
            points_for_arm = num_arm_points + (1 if arm_index < remainder else 0)

            for _ in range(points_for_arm):  # Generate the specified number of points for each arm
                # theta = np.random.triangular(0, 0, np.pi/2)  # Random angle
                theta = self.pdf(distribution="uniform", params={"low": 0, "high": 1})
                r = self.swirling_uni_pdf(theta, arm_index, max_distance)

                # Calculate width effect based on distance
                width_effect = sigma * (1 - (r / max_distance))  # Linearly decrease width

                # Convert to Cartesian coordinates
                x_center = r * np.cos(theta + arm_index * (2 * np.pi / arm_size))
                y_center = r * np.sin(theta + arm_index * (2 * np.pi / arm_size))

                # Generate random offsets based on the current width
                x_offset = self.pdf(distribution="uniform", params={"low": -width_effect / 2, "high": width_effect / 2})
                y_offset = self.pdf(distribution="uniform", params={"low": -width_effect / 2, "high": width_effect / 2})

                # Add offsets to the center point
                x_new = x_center + x_offset
                y_new = y_center + y_offset

                # Add to the lists
                X.append((x_new, y_new))

        return np.array(X)

    def generate(self):
        X = self.generate_uni_star_points()
        X = self.normalize(X)
        self.data = X
        return X