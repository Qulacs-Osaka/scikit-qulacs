import numpy as np


def get_angles(x: np.ndarray) -> np.ndarray:
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def preprocess_input(x: np.ndarray) -> np.ndarray:
    padding = 0.3 * np.ones((len(x), 1))
    x_pad = np.c_[np.c_[x, padding], np.zeros((len(x), 1))]

    # normalize each input
    normalization = np.sqrt(np.sum(x_pad**2, -1))
    x_norm = (x_pad.T / normalization).T

    # angles for state preparation are new features
    return np.array([get_angles(x) for x in x_norm])
