from sklearn.datasets import make_s_curve
from .base_dataset_class import BaseDataset2D

class SCurveDataset(BaseDataset2D):
    def __init__(self, num_samples=10000, dataset_name ='S_Curve', random_state=42, noise_level = 0.1):
        super().__init__(num_samples, dataset_name, random_state)
        self.noise_level = noise_level
        
    def generate(self):
        X, _ = make_s_curve(self.num_samples, noise=self.noise_level, random_state=self.random_state)
        
        # Project to 2D by taking the first two dimensions
        X = X[:,[0,2]]
        
        X = self.normalize(X)
        self.data = X
        return X
