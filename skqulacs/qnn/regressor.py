from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import Observable
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.solver import Solver


@dataclass(eq=False)
class QNNRegressor:
    """Class to solve regression problems with quantum neural networks
    The output is taken as expectation values of pauli Z operator acting on the first qubit, i.e., output is <Z_0>.

    Args:
        circuit: Circuit to use in the learning.
        solver: Solver to use(Nelder-Mead is not recommended).
        n_output: Dimentionality of each output data.
        cost: Cost function. MSE only for now. MSE computes squared sum after normalization.
        do_x_scale: Whether to scale x.
        do_x_scale: Whether to scale y.
        x_norm_range: Normalize x in [+-xy_norm_range].
        y_norm_range: Normalize y in [+-y_norm_range]. Setting y_norm_range to 0.7 improves performance.

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

    circuit: LearningCircuit
    solver: Solver
    cost: Literal["mse"] = field(default="mse")
    do_x_scale: bool = field(default=True)
    do_y_scale: bool = field(default=True)
    x_norm_range: float = field(default=1.0)
    y_norm_range: float = field(default=0.7)

    observables: List[Observable] = field(init=False, default_factory=list)
    n_qubit: int = field(init=False)
    n_outputs: int = field(init=False)
    x_scaler: MinMaxScaler = field(init=False)
    y_scaler: MinMaxScaler = field(init=False)

    def __post_init__(self) -> None:
        self.n_qubit = self.circuit.n_qubit

        if self.do_x_scale:
            self.scale_x_scaler = MinMaxScaler(
                feature_range=(-self.x_norm_range, self.x_norm_range)
            )
        if self.do_y_scale:
            self.scale_y_scaler = MinMaxScaler(
                feature_range=(-self.y_norm_range, self.y_norm_range)
            )

    def fit(
        self,
        x_train: NDArray[np.float_],
        y_train: NDArray[np.float_],
        maxiter: Optional[int] = None,
    ) -> Tuple[float, List[float]]:
        """
        Args:
            x_list: List of training data inputs whose shape is (n_sample, n_features).
            y_list: List of training data outputs whose shape is (n_sample, n_output_dims).
            maxiter: The number of iterations to pass scipy.optimize.minimize
        Returns:
            loss: Loss after learning.
            theta: Parameter theta after learning.
        """

        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        if y_train.ndim == 1:
            y_train = y_train.reshape((-1, 1))

        if self.do_x_scale:
            x_scaled = self.scale_x_scaler.fit_transform(x_train)
        else:
            x_scaled = x_train

        if self.do_y_scale:
            y_scaled = self.scale_y_scaler.fit_transform(y_train)
        else:
            y_scaled = y_train

        self.n_outputs = y_scaled.shape[1]

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
