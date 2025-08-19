import numpy as np
from .base_dataset_class import BaseDataset2D

class GMMDataset(BaseDataset2D):
    """
    GMM dataset in the same template as EightGaussiansDataset.
    Defaults mirror your energy setup: 40 modes, loc_scaling=40.
    Variance is diagonal with exp(log_var_scaling).
    """
    def __init__(
        self,
        num_samples=10000,
        dataset_name='GMM',
        random_state=42,
        n_mixes=40,
        loc_scaling=40.0,
        log_var_scaling=1.0,   # variance = exp(log_var_scaling)
    ):
        super().__init__(num_samples, dataset_name, random_state)
        self.n_mixes = n_mixes
        self.loc_scaling = float(loc_scaling)
        self.log_var_scaling = float(log_var_scaling)

    def pdf(self, distribution, params, size):
        if distribution == "multivariate_normal":
            return self.random_multivariate_normal(params["mean"], params["covariance"], size)

    def generate(self):
        # number of points per component (like EightGaussiansDataset)
        n_points = int(self.num_samples / self.n_mixes)

        rng = np.random.default_rng(self.random_state)

        # Means sampled i.i.d. ~ N(0, loc_scaling^2 I) in R^2 (matches the "random centers" idea)
        means = rng.normal(loc=0.0, scale=self.loc_scaling, size=(self.n_mixes, 2))

        # Diagonal covariance shared across components: variance = exp(log_var_scaling)
        var = float(np.exp(self.log_var_scaling))
        covariance = var * np.eye(2)

        # Allocate and sample per-component (same pattern as EightGaussiansDataset)
        X = np.zeros((self.n_mixes * n_points, 2))
        for i, mean in enumerate(means):
            X[i * n_points:(i + 1) * n_points] = self.pdf(
                distribution="multivariate_normal",
                params={"mean": mean, "covariance": covariance},
                size=n_points,
            )

        # normalize to dataset’s expected range
        X = self.normalize(X)
        self.data = X
        return X