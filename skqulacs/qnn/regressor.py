from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import Observable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.solver import Solver


class QNNRegressor:
    """Class to solve regression problems with quantum neural networks
    The output is taken as expectation values of pauli Z operator acting on the first qubit, i.e., output is <Z_0>.
    Examples:
        >>> from skqulacs.qnn import QNNRegressor
        >>> from skqulacs.circuit import create_qcl_ansatz
        >>> n_qubits = 4
        >>> depth = 3
        >>> evo_time = 0.5
        >>> circuit = create_qcl_ansatz(n_qubits, depth, evo_time)
        >>> model = QNNRegressor(circuit)
        >>> _, theta = model.fit(x_train, y_train, maxiter=1000)
        >>> x_list = np.arange(x_min, x_max, 0.02)
        >>> y_pred = qnn.predict(theta, x_list)
    """

    def __init__(
        self,
        circuit: LearningCircuit,
        solver: Solver,
        cost: Literal["mse"] = "mse",
        do_x_scale: bool = True,
        do_y_scale: bool = True,
        x_norm_range: float = 1.0,
        y_norm_range: float = 0.7,
    ) -> None:
        """
        :param circuit: Circuit to use in the learning.
        :param solver: Solver to use(Nelder-Mead is not recommended).
        :param cost: Cost function. MSE only for now. MSE computes squared sum after normalization.
        :param do_x_scale: Whether to scale x.
        :param do_x_scale: Whether to scale y.
        :param y_norm_range: Normalize y in [+-y_norm_range].
        :param callback: Callback function. Available only with Adam.
        Setting y_norm_range to 0.7 improves performance.
        :param tol: use n_iter_no_change
        :param n_iter_no_change: (cost reduce < tol) continues n_iter_no_change times -> stopping (Adam only)

        """
        self.n_qubit = circuit.n_qubit
        self.circuit = circuit
        self.solver = solver
        self.cost = cost
        self.do_x_scale = do_x_scale
        self.do_y_scale = do_y_scale
        self.x_norm_range = x_norm_range
        self.y_norm_range = y_norm_range
        self.observables = []

    def fit(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        maxiter: Optional[int] = None,
    ) -> Tuple[float, List[float]]:
        """
        :param x_list: List of x to fit.
        :param y_list: List of y to fit.
        :param maxiter: The number of iterations to pass scipy.optimize.minimize
        :return: Loss after learning.
        :return: Parameter theta after learning.
        """

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))

        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        if self.do_y_scale:
            self.scale_y_scaler = MinMaxScaler(
                feature_range=(-self.y_norm_range, self.y_norm_range)
            )
            y_scaled = self.scale_y_scaler.fit_transform(y_train)
        else:
            y_scaled = y_train

        if y_train.ndim == 2:
            self.n_outputs = len(y_scaled[0])
        else:
            self.n_outputs = 1

        self.observables = []
        for i in range(self.n_outputs):
            observable = Observable(self.n_qubit)
            observable.add_operator(1.0, f"Z {i}")
            self.observables.append(observable)

        theta_init = self.circuit.get_parameters()
        return self.solver.run(
            self.cost_func,
            self._cost_func_grad,
            theta_init,
            x_scaled,
            y_scaled,
            maxiter,
        )

    def predict(self, x_test: NDArray[np.float_]) -> NDArray[np.float_]:
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape((-1, 1))
        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.transform(x_test)
        else:
            x_scaled = x_test

        if self.do_y_scale:
            y_pred = self.scale_y_scaler.inverse_transform(
                self._predict_inner(x_scaled)
            )
        else:
            y_pred = self._predict_inner(x_scaled)

        return y_pred

    def _predict_inner(self, x_scaled: NDArray[np.float_]) -> NDArray[np.float_]:
        res = []
        for x in x_scaled:
            state = self.circuit.run(x)
            r = [
                self.observables[i].get_expectation_value(state)
                for i in range(self.n_outputs)
            ]
            res.append(r)
        return np.array(res)

    def cost_func(
        self,
        theta: List[float],
        x_scaled: NDArray[np.float_],
        y_scaled: NDArray[np.float_],
    ) -> float:
        if self.cost == "mse":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)

            cost = mean_squared_error(y_pred, y_scaled)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _cost_func_grad(
        self,
        theta: List[float],
        x_scaled: NDArray[np.float_],
        y_scaled: NDArray[np.float_],
    ) -> NDArray[np.float_]:
        self.circuit.update_parameters(theta)

        mto = self._predict_inner(x_scaled).copy()

        grad = np.zeros(len(theta))

        for h in range(len(x_scaled)):
            backobs = Observable(self.n_qubit)
            if self.n_outputs >= 2:
                for i in range(self.n_outputs):
                    backobs.add_operator(
                        (-y_scaled[h][i] + mto[h][i]) / self.n_outputs, f"Z {i}"
                    )
            else:
                backobs.add_operator((-y_scaled[h] + mto[h][0]) / self.n_outputs, "Z 0")
            grad += self.circuit.backprop(x_scaled[h], backobs)

        grad /= len(x_scaled)
        return grad
