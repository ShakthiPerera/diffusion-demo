from scipy.stats import kurtosis
import torch
import matplotlib.pyplot as plt
import numpy as np


def whiteness_measures(data):
    cov_matrix = np.cov(data, rowvar=False)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    isotropy_ratio = np.max(eigenvalues) / np.min(eigenvalues)

    alpha = np.mean(eigenvalues)
    identity_matrix = np.eye(cov_matrix.shape[0])
    frobenius_norm_diff = np.linalg.norm(cov_matrix - identity_matrix, ord="fro")

    return isotropy_ratio, frobenius_norm_diff


def kurtosis_(data):
    kurtosis_ = kurtosis(data, fisher=True)
    return kurtosis_


def isotropy(data):
    iso = torch.trace((data.T @ data)) / data.size(dim=0)
    return iso


def metric_plot(
    metric_values, metric_name, ylim, save_name, path = ".", smoothed=False, window_size=50,
):
    plt.figure(figsize=(8, 6))

    # Define the window size for the moving average
    window_size = window_size

    # Compute the moving average
    if smoothed:
        smoothed_metric = np.convolve(
            metric_values, np.ones(window_size) / window_size, mode="same"
        )
        smoothed_metric[:window_size] = metric_values[:window_size]
        plt.plot(smoothed_metric, label=f"{metric_name}")
    else:
        plt.plot(metric_values, label=f"{metric_name}")
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel(f"{metric_name} Values")
    plt.title(f"{metric_name} Plot ")
    if not ylim is None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.savefig(f"{path}/{save_name}.png")


def kurtosis_plot(metric_values, metric_name, ylim, save_name, path = "."):
    plt.figure(figsize=(8, 6))

    # Define the window size for the moving average
    window_size = 10

    # Compute the moving average
    smoothed_metric = np.convolve(
        metric_values, np.ones(window_size) / window_size, mode="same"
    )
    smoothed_metric[:window_size] = metric_values[:window_size]
    plt.scatter(range(1000), smoothed_metric, label=f"{metric_name}", s=1)
    plt.axhline(y=0, color="red", linestyle="-")  # Add a red line along y = 0

    # Label the region above the x-axis as "Super Gaussian" and below the x-axis as "Sub Gaussian"
    plt.text(
        900,
        1.5,
        "Super Gaussian",
        horizontalalignment="center",
        color="black",
        fontsize=12,
    )
    plt.text(
        900,
        -1.5,
        "Sub Gaussian",
        horizontalalignment="center",
        color="black",
        fontsize=12,
    )

    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel(f"{metric_name} Values")
    plt.title(f"{metric_name} Plot ")
    if not ylim is None:
        plt.ylim(*ylim)
    plt.grid(True)
    plt.savefig(f"{path}/{save_name}.png")

