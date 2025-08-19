import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


class BaseDataset2D:
    def __init__(self, num_samples=10000, dataset_name="", random_state=42):
        self.num_samples = num_samples
        self.random_state = random_state
        self.dataset_name = dataset_name
        self.data = None  # To store the generated dataset
        self.rng = np.random.default_rng(self.random_state)

    def normalize(self, X):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        return scaler.fit_transform(X)

    def generate(self):
        pass

    def random_uniform(self, low, high, size=None):
        return self.rng.uniform(low, high, size)

    def random_normal(self, mean, std, size=None):
        return self.rng.normal(mean, std, size)

    def random_exponential(self, scale, size=None):
        return self.rng.exponential(scale, size)

    def random_multivariate_normal(self, mean, covariance, size=None):
        return self.rng.multivariate_normal(mean, covariance, size)

    def random_standard_normal(self, size=None):
        return self.rng.standard_normal(size)

    def random_shuffle(self, data):
        return self.rng.shuffle(data)

    def plot_dataset(self, contours=True):
        if self.data is None:
            print("No dataset available. Generate a dataset first")
            return
        if contours:
            sns.kdeplot(
                x=self.data[:, 0],
                y=self.data[:, 1],
                fill=False,
                levels=[0.1, 0.2, 0.4, 0.6, 0.8],
                gridsize=150,
                cut=3,
                color="blue",
                alpha=0.8,
            )
        plt.scatter(self.data[:, 0], self.data[:, 1], s=2, color="coral", alpha=1.0)
        plt.title(f"Scatter Plot of {self.dataset_name} Rescaled to [-1, 1]")
        plt.xlabel(" ")
        plt.ylabel(" ")
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.grid(True)
        plt.savefig("plot.png")
        plt.show()
