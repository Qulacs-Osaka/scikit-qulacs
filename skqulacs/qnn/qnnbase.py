from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class QNN(ABC):
    @abstractmethod
    def fit(
        self, x_train: List[List[float]], y_train: List[float], maxiter: Optional[int]
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
