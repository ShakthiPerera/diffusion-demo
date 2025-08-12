import numpy as np
from .base_dataset_class import BaseDataset2D

class MultimodalGasussiansDataset(BaseDataset2D):
    
    def __init__(self, num_samples=10000, dataset_name='Multimodal_Gaussians', random_state=42):
         super().__init__(num_samples, dataset_name, random_state)

    def pdf(self, distribution, params, size=None):
        if distribution == "normal":
            return self.random_normal(params["mean"], params["std"], size)
    
    def generate_CombinedGaussian_distributions(self):
        """
        Generate points from Gaussian, Sub-Gaussian, and Super-Gaussian distributions within -1 and 1 range.
        """
        X = []
        # Gaussian (centered at origin, narrow spread)
        mean_gaussian_x = 0
        mean_gaussian_y = 0
        std_gaussian_x = 0.075
        std_gaussian_y = 0.075

        # Sub-Gaussian (slightly shifted to the left, moderate spread)
        mean_subgaussian_x = -0.25
        mean_subgaussian_y = 0.25
        std_subgaussian_x = 0.1
        std_subgaussian_y = 0.1

        # Super-Gaussian (slightly shifted to the right, wider spread)
        mean_supergaussian_x = 0.25
        mean_supergaussian_y = -0.25
        std_supergaussian_x = 0.15
        std_supergaussian_y = 0.15

        # Generate points for each distribution
        x_gaussian = self.pdf(distribution="normal", params={"mean": mean_gaussian_x, "std": std_gaussian_x}, size= self.num_samples//3)
        y_gaussian = self.pdf(distribution="normal", params={"mean": mean_gaussian_y, "std": std_gaussian_y}, size=self.num_samples//3)

        x_subgaussian = self.pdf(distribution="normal", params={"mean": mean_subgaussian_x, "std": std_subgaussian_x}, size=self.num_samples//3)
        y_subgaussian = self.pdf(distribution="normal", params={"mean": mean_subgaussian_y, "std": std_subgaussian_y}, size=self.num_samples//3)

        x_supergaussian = self.pdf(distribution="normal", params={"mean": mean_supergaussian_x, "std": std_supergaussian_x}, size=self.num_samples-2*self.num_samples//3)
        y_supergaussian = self.pdf(distribution="normal", params={"mean": mean_supergaussian_y, "std": std_supergaussian_y}, size=self.num_samples-2*self.num_samples//3)

        # Combine all points
        x_combined1 = np.concatenate([x_gaussian, x_subgaussian, x_supergaussian])
        y_combined1 = np.concatenate([y_gaussian, y_subgaussian, y_supergaussian])

        # Rescale points to the range of -1 to +1
        combined_scaled = self.normalize(np.vstack((x_combined1, y_combined1)).T)

        # Separate x and y coordinates after scaling
        x_combined, y_combined = combined_scaled[:, 0], combined_scaled[:, 1]

        X.append((x_combined, y_combined))
        return np.array(X).reshape(-1, 2)
    
    def generate(self):
        X = self.generate_crescent_with_circles()
        self.data = X
        return X