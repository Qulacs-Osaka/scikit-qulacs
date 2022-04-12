from __future__ import annotations

from typing import List, Optional

import numpy as np
from qulacs import Observable
from scipy.special import softmax
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN
from skqulacs.qnn.optimizer import Optimizer


class QNNClassifier(QNN):
    """Class to solve classification problems by quantum neural networks
    The prediction is made by making a vector which predicts one-hot encoding of labels.
    The prediction is made by
    1. taking expectation values of Pauli Z operator of each qubit <Z_i>,
    2. taking softmax function of the vector (<Z_0>, <Z_1>, ..., <Z_{n-1}>).
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

    def __init__(
        self,
        circuit: LearningCircuit,
        num_class: int,
        solver: Optimizer,
        cost: Literal["log_loss"] = "log_loss",
        do_x_scale: bool = True,
        y_exp_ratio=5.0,
    ) -> None:
        """
        :param circuit: Circuit to use in the learning.
        :param num_class: The number of classes; the number of qubits to measure. must be n_qubits >= num_class .
        :param solver: Solver to use(Nelder-Mead is not recommended).
        :param cost: Cost function. log_loss only for now.
        :param do_x_scale: Whether to scale x.
        :param y_exp_ratio:
            coeffcient used in the application of softmax function.
            the output prediction vector is made by transforming (<Z_0>, <Z_1>, ..., <Z_{n-1}>)
            to (y_1, y_2, ..., y_(n-1)) where y_i = e^{<Z_i>*y_exp_scale}/(sum_j e^{<Z_j>*y_exp_scale})
        :param callback: Callback function. Available only with Adam.
        """

        self.n_qubit = circuit.n_qubit
        self.circuit = circuit
        self.num_class = num_class
        self.solver = solver
        self.cost = cost
        self.do_x_scale = do_x_scale
        self.x_norm_range = x_norm_range
        self.y_exp_ratio = y_exp_ratio
        self.scale_x_param = []
        self.scale_y_param = []
        self.observables = [Observable(self.n_qubit) for _ in range(self.n_qubit)]
        for i in range(self.n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[int],
        maxiter: Optional[int] = None,
    ):
        """
        :param x_list: List of training data inputs.
        :param y_list: List of labels to fit. ;Labels must be represented as integers.
        :param maxiter: The number of iterations to pass scipy.optimize.minimize
        :return: Loss after learning.
        :return: Parameter theta after learning.
        """

        y_scaled = y_train
        x_train = np.array(x_train)
        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        y_scaled = self.do_y_scale(y_train)
        theta_init = self.circuit.get_parameters()
        self.solver.setup(self.cost_func, self._cost_func_grad)
        loss, theta_opt = self.solver.run(theta_init, x_scaled, y_scaled, maxiter)
        # TODO: Extract solvers if the same one is used for classifier and regressor.

        return loss, theta_opt

    def predict(self, x_test: List[List[float]]):
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        x_test = np.array(x_test)
        if x_test.ndim == 1:
            x_test = x_test.reshape((-1, 1))
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test

        y_pred = self._predict_inner(x_scaled).argmax(axis=1)
        return y_pred

    def _predict_inner(self, x_list):
        res = np.zeros((len(x_list), self.num_class))
        for i in range(len(x_list)):
            state = self.circuit.run(x_list[i])
            for j in range(self.num_class):
                res[i][j] = (
                    self.observables[j].get_expectation_value(state) * self.y_exp_ratio
                )
        return res

    # TODO: Extract cost function to outer class to accept other type of ones.
    def cost_func(self, theta, x_scaled, y_scaled):
        if self.cost == "log_loss":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)
            y_pred_sm = softmax(y_pred, axis=1)
            return log_loss(y_scaled, y_pred_sm)
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _cost_func_grad(self, theta, x_scaled, y_scaled):
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

        self.circuit.update_parameters(theta)
        grad /= len(x_scaled)
        return grad
