from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class QNN(ABC):
    @abstractmethod
    def fit(
        self, x_train: List[float], y_train: List[float], maxiter: int
    ) -> Tuple[float, np.ndarray]:
        pass

    @abstractmethod
    def predict(self, theta: List[float], x_list: List[float]) -> float:
        pass
