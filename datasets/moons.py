from sklearn.datasets import make_moons
from .base_dataset_class import BaseDataset2D

class MoonDataset(BaseDataset2D): 
    def __init__(self, num_samples=10000, dataset_name ='Moons', random_state=42, noise_level=0.1):
        super().__init__(num_samples, dataset_name, random_state)
        self.noise_level = noise_level
        
    def generate(self):
        X, _ = make_moons(self.num_samples, noise=self.noise_level, random_state=self.random_state)
        X = self.normalize(X)
        self.data = X
        return X