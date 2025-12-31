import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
from scipy.stats import multivariate_normal
from math import pi, gamma


def knn_entropy_kozachenko_leonenko(
    X: np.ndarray,
    k: int = 5,
    base: float = np.e,
    metric: str = "chebyshev",
    eps: float = 1e-12,
) -> float:
    """
    Kozachenko–Leonenko kNN differential entropy estimator.

    Parameters
    ----------
    X : (n, d) array
        Continuous samples.
    k : int
        k-th nearest neighbor (k>=1). Typical: 3..10.
    base : float
        Log base. np.e -> nats, 2 -> bits.
    metric : str
        'chebyshev' (recommended, matches unit ball constant easily),
        or 'euclidean'. (Chebyshev is standard in many KL implementations.)
    eps : float
        Small positive number to avoid log(0).

    Returns
    -------
    h : float
        Estimated differential entropy in units of log(base).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, d).")
    n, d = X.shape
    if n <= k:
        raise ValueError(f"Need n > k. Got n={n}, k={k}.")
    if k < 1:
        raise ValueError("k must be >= 1.")

    # Lazy import so code is copy-paste friendly.
    from sklearn.neighbors import NearestNeighbors
    from scipy.special import digamma
    from math import pi, gamma

    # Fit neighbors; n_neighbors=k+1 because point is its own nearest neighbor.
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X, return_distance=True)

    # distance to k-th neighbor (skip index 0 = self, then take k)
    eps_k = distances[:, k]
    eps_k = np.maximum(eps_k, eps)

    # Unit ball volume c_d depends on metric.
    # For Chebyshev (L_infty): unit ball is a hypercube of side length 2 => volume = 2^d.
    # For Euclidean (L2): volume = pi^(d/2) / Gamma(d/2 + 1).
    if metric.lower() in ["chebyshev", "l_inf", "linf", "infinity"]:
        c_d = 2.0 ** d
    elif metric.lower() in ["euclidean", "l2"]:
        c_d = (pi ** (d / 2.0)) / gamma(d / 2.0 + 1.0)
    else:
        raise ValueError("metric must be 'chebyshev' or 'euclidean' for correct c_d.")

    # KL estimator (one of the standard forms)
    h_nats = (
        digamma(n)
        - digamma(k)
        + np.log(c_d)
        + (d / n) * np.sum(np.log(eps_k))
    )

    # Convert to requested base
    return float(h_nats / np.log(base))


def kde_entropy(
    X: np.ndarray,
    bandwidth: float = None,
    base: float = np.e,
    eps: float = 1e-300,
) -> float:
    """
    KDE plug-in estimator of differential entropy:
        h ≈ - (1/n) Σ log p_hat(x_i)

    Notes:
    - This is biased; bandwidth choice dominates results.
    - For stability, prefer cross-validated bandwidth or a principled rule.

    Parameters
    ----------
    X : (n, d) array
        Continuous samples.
    bandwidth : float or None
        KDE bandwidth. If None, uses Scott's rule (simple default).
    base : float
        Log base. np.e -> nats, 2 -> bits.
    eps : float
        Floor for density to avoid log(0).

    Returns
    -------
    h : float
        Estimated differential entropy in units of log(base).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n, d).")
    n, d = X.shape

    from sklearn.neighbors import KernelDensity

    if bandwidth is None:
        # Scott's rule: h = n^{-1/(d+4)} * std (rough heuristic)
        std = np.std(X, axis=0, ddof=1)
        scale = float(np.mean(std))
        bandwidth = (n ** (-1.0 / (d + 4.0))) * (scale if scale > 0 else 1.0)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X)

    log_p = kde.score_samples(X)  # log density at samples
    # Floor density via log-space clamp
    log_p = np.maximum(log_p, np.log(eps))

    h_nats = -float(np.mean(log_p))
    return h_nats / np.log(base)




def kl_data_to_gaussian_knn(X, k=5, metric="chebyshev"):
    """
    Estimate KL(p_data || N(mu, Sigma)) using kNN.
    X : (n,2) data points
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert d == 2

    # Fit Gaussian to data
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)
    gauss = multivariate_normal(mean=mu, cov=Sigma)

    # kNN distances in data
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    eps = distances[:, k]

    # Unit ball volume
    if metric == "chebyshev":
        Vd = 2.0 ** d
    else:  # euclidean
        Vd = (pi ** (d / 2)) / gamma(d / 2 + 1)

    # log p_data(x) estimate
    log_p_data = (
        digamma(k)
        - digamma(n)
        - np.log(Vd)
        - d * np.log(eps)
    )

    # log p_gauss(x) exact
    log_p_gauss = gauss.logpdf(X)

    # KL estimate
    kl = np.mean(log_p_data - log_p_gauss)
    return float(kl)




def mutual_information_knn(X, k=5):
    """
    Kraskov kNN mutual information estimator for 2D data.
    X : (n,2)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    assert d == 2

    x = X[:, 0][:, None]
    y = X[:, 1][:, None]

    # Joint space
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(X)
    dist_joint, _ = nn_joint.kneighbors(X)
    eps = dist_joint[:, k]

    # Marginal counts
    nn_x = NearestNeighbors(metric="chebyshev").fit(x)
    nn_y = NearestNeighbors(metric="chebyshev").fit(y)

    nx = np.array([
        nn_x.radius_neighbors([x[i]], eps[i] - 1e-12, return_distance=False)[0].size - 1
        for i in range(n)
    ])

    ny = np.array([
        nn_y.radius_neighbors([y[i]], eps[i] - 1e-12, return_distance=False)[0].size - 1
        for i in range(n)
    ])

    mi = (
        digamma(k)
        + digamma(n)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )

    return float(mi)




if __name__ == "__main__":
    # Example usage (2D)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2)) @ np.array([[1.0, 0.7], [0.0, 0.5]]).T  # correlated Gaussian

    h_knn = knn_entropy_kozachenko_leonenko(X, k=5, metric="chebyshev", base=np.e)
    h_kde = kde_entropy(X, bandwidth=None, base=np.e)

    print("kNN entropy (nats):", h_knn)
    print("KDE entropy (nats):", h_kde)
