from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional, Tuple
from qulacs.gate import X, Z, DenseMatrix
from numpy.random import Generator, default_rng
import numpy as np


# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


def _get_x_scale_param(x):
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    sa = (maximum - minimum) / 2
    return [minimum, maximum, sa]


def _min_max_scaling(x: List[List[float]], scale_x_param):
    """[-1, 1]の範囲に規格化"""
    # print([((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x])
    return [((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x]


def _softmax(x):
    """softmax function
    :param x: ndarray
    """
    y = np.exp(x) / np.sum(np.exp(x))
    return y


class QNN(ABC):
    @abstractmethod
    def fit(
        self, x_train: List[List[float]], y_train, maxiter: Optional[int]
    ) -> Tuple[float, np.ndarray]:
        """Fit the model to given train data.

        Args:
            x_train: Train data of independent variable.
            y_train: Train data of dependent variable.
            maxiter: Maximum number of iterations for a cost minimization solver.

        Returns:
            loss: Loss of minimized cost function.
            theta_opt: Parameter of optimized model.
        """
        pass

    @abstractmethod
    def predict(self, x_test: List[List[float]]) -> List[float]:
        """Predict outcome for given data.

        Args:
            theta: Parameter of model. For most cases, give `theta_opt` from `QNN.fit`.
            x_list: Input data to predict outcome.

        Returns:
            y_pred: List of predicted data. `y_pred[i]` corresponds to `x_list[i]`.
        """
        pass
