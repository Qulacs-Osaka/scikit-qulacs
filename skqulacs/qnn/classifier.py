from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import Observable
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.solver import Solver


@dataclass(eq=False)
class QNNClassifier:
    """Class to solve classification problems by quantum neural networks
    The prediction is made by making a vector which predicts one-hot encoding of labels.
    The prediction is made by
    1. taking expectation values of Pauli Z operator of each qubit <Z_i>,
    2. taking softmax function of the vector (<Z_0>, <Z_1>, ..., <Z_{n-1}>).

    Args:
        circuit: Circuit to use in the learning.
        num_class: The number of classes; the number of qubits to measure. must be n_qubits >= num_class .
        solver: Solver to use(Nelder-Mead is not recommended).
        cost: Cost function. log_loss only for now.
        do_x_scale: Whether to scale x.
        y_exp_ratio:
            coeffcient used in the application of softmax function.
            the output prediction vector is made by transforming (<Z_0>, <Z_1>, ..., <Z_{n-1}>)
            to (y_1, y_2, ..., y_(n-1)) where y_i = e^{<Z_i>*y_exp_scale}/(sum_j e^{<Z_j>*y_exp_scale})

    Examples:
        >>> from skqulacs.qnn import QNNClassifier
        >>> from skqulacs.circuit import create_qcl_ansatz
        >>> n_qubits = 4
        >>> depth = 3
        >>> evo_time = 0.5
        >>> circuit = create_qcl_ansatz(n_qubits, depth, evo_time)
        >>> model = QNNRClassifier(circuit)
        >>> _, theta = model.fit(x_train, y_train, maxiter=1000)
        >>> x_list = np.arange(x_min, x_max, 0.02)
        >>> y_pred = qnn.predict(theta, x_list)
    """

    circuit: LearningCircuit
    num_class: int
    solver: Solver
    cost: Literal["log_loss"] = field(default="log_loss")
    do_x_scale: bool = field(default=True)
    x_norm_range: float = field(default=1.0)
    y_exp_ratio: float = field(default=2.2)

    observables: List[Observable] = field(init=False)
    n_qubit: int = field(init=False)
    x_scaler: MinMaxScaler = field(init=False)

    def __post_init__(self) -> None:
        self.n_qubit = self.circuit.n_qubit
        self.observables = [Observable(self.n_qubit) for _ in range(self.n_qubit)]
        for i in range(self.n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )

    def fit(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.int_],
        maxiter: Optional[int] = None,
    ) -> Tuple[float, List[float]]:
        """
        Args:
            x_train: List of training data inputs whose shape is (n_sample, n_features).
            y_train: List of labels to fit. Labels must be represented as integers. Shape is (n_samples,)
            maxiter: The number of maximum iterations to pass scipy.optimize.minimize
        Returns:
            loss: Loss after learning.
            theta: Parameter theta after learning.
        """

        y_scaled = y_train
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        theta_init = self.circuit.get_parameters()
        return self.solver.run(
            self.cost_func,
            self._cost_func_grad,
            theta_init,
            x_scaled,
            y_scaled,
            maxiter,
        )

    def predict(self, x_test: NDArray[np.float_]) -> NDArray[np.int_]:
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome whose shape is (n_samples,).
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape((-1, 1))
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test

        y_pred: NDArray[np.int_] = self._predict_inner(x_scaled).argmax(axis=1)
        return y_pred

    def _predict_inner(self, x_list: NDArray[np.float_]) -> NDArray[np.float_]:
        res = np.zeros((len(x_list), self.num_class))
        for i in range(len(x_list)):
            state = self.circuit.run(x_list[i])
            for j in range(self.num_class):
                res[i][j] = (
                    self.observables[j].get_expectation_value(state) * self.y_exp_ratio
                )
        return res

    # TODO: Extract cost function to outer class to accept other type of ones.
    def cost_func(
        self,
        theta: List[float],
        x_scaled: NDArray[np.float_],
        y_scaled: NDArray[np.int_],
    ) -> float:
        if self.cost == "log_loss":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)
            y_pred_sm = softmax(y_pred, axis=1)
            return log_loss(y_scaled, y_pred_sm)
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _cost_func_grad(
        self,
        theta: List[float],
        x_scaled: NDArray[np.float_],
        y_scaled: NDArray[np.int_],
    ) -> NDArray[np.float_]:
        self.circuit.update_parameters(theta)

        y_pred = self._predict_inner(x_scaled)
        y_pred_sm = softmax(y_pred, axis=1)
        grad = np.zeros(len(theta))
        for sample_index in range(len(x_scaled)):
            backobs = Observable(self.n_qubit)
            for current_class in range(self.num_class):
                expected = 0.0
                if current_class == y_scaled[sample_index]:
                    expected = 1.0
                backobs.add_operator(
                    (-expected + y_pred_sm[sample_index][current_class])
                    * self.y_exp_ratio,
                    f"Z {current_class}",
                )
            grad += self.circuit.backprop(x_scaled[sample_index], backobs)

        grad /= len(x_scaled)
        return grad
