from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class QNN(ABC):
    @abstractmethod
    def fit(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.int_],
        maxiter: Optional[int] = None,
    ) -> Tuple[float, List[float]]:
        """Fit the model to given train data.

        Args:
            x_train: Train data of independent variable whose shape is (n_samples, n_features).
            y_train: Train data of dependent variable whose shape is (n_samples).
            maxiter: Maximum number of iterations for a cost minimization solver.

        Returns:
            loss: Loss of minimized cost function.
            theta_opt: Parameter of optimized model.
        """
        pass

    @abstractmethod
    def predict(self, x_test: NDArray[np.float_]) -> NDArray[np.int_]:
        """Predict outcome for given data.

        Args:
            x_list: Input data to predict outcome.

        Returns:
            y_pred: List of predicted data. `y_pred[i]` corresponds to `x_list[i]`.
        """
        pass
