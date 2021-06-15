from abc import ABC, abstractmethod


class QNN(ABC):
    @abstractmethod
    def fit(self, x_train, y_train, maxiter: int) -> Tuple[float, np.ndarray]:
        pass

    @abstractmethod
    def predict(self, theta, x) -> float:
        pass
