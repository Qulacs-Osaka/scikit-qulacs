from typing import List

import numpy as np
from numpy.typing import NDArray
from qulacs import QuantumState
from qulacs.state import inner_product
from scipy.stats import loguniform
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV

from skqulacs.circuit import LearningCircuit


class QKRR:
    """class to solve regression problems with kernel ridge regressor with a quantum kernel"""

    def __init__(self, circuit: LearningCircuit, n_iteration=10) -> None:
        """
        :param circuit: circuit to generate quantum feature
        """
        self.krr = KernelRidge(kernel="precomputed")
        self.kernel_ridge_tuned = None
        self.circuit = circuit
        self.data_states: List[QuantumState] = []
        self.n_qubit = 0
        self.n_iteration = n_iteration

    def fit(self, x: NDArray[np.float_], y: NDArray[np.int_]) -> None:
        """
        train the machine.
        :param x: training inputs
        :param y: training teacher values
        """
        print(y)
        self.n_qubit = len(x[0])
        kar = np.zeros((len(x), len(x)))
        # Compute UΦx to get kernel of `x` and `y`.
        for i in range(len(x)):
            self.data_states.append(self.circuit.run(x[i]))

        for i in range(len(x)):
            for j in range(len(x)):
                kar[i][j] = (
                    abs(inner_product(self.data_states[i], self.data_states[j])) ** 2
                )

        self.krr.fit(kar, y)

        # hyperparameter tuning
        alpha_low = 1e-3
        alpha_high = 1e2
        n_iteration = 5
        random_state = 0
        param_distributions = {
            "alpha": loguniform(
                alpha_low, alpha_high
            ),  # Hyperparameter in the cost function for the regularizaton
            # "kernel__length_scale": loguniform(1e-3, 1e3), # Hyperparameter of the Kernel (If we apply the Quantum Kernel, this must be ignored)
            # "kernel__periodicity": loguniform(1e0, 1e1), # For periodic Kernel
        }
        kernel_ridge_tuned = RandomizedSearchCV(
            self.krr,
            param_distributions=param_distributions,
            n_iter=n_iteration,
            random_state=random_state,
        )

        kernel_ridge_tuned.fit(kar, y)
        print(kernel_ridge_tuned.best_params_)
        self.kernel_ridge_tuned = kernel_ridge_tuned

    def predict(self, xs: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        predict y values for each of xs
        :param xs: inputs to make predictions
        :return: List[int], predicted values of y
        """
        kar = np.zeros((len(xs), len(self.data_states)))
        for i in range(len(xs)):
            x_qc = self.circuit.run(xs[i])
            for j in range(len(self.data_states)):
                kar[i][j] = abs(inner_product(x_qc, self.data_states[j])) ** 2
        predicted: NDArray[np.float_] = self.kernel_ridge_tuned.predict(kar)
        return predicted
