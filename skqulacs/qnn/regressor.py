from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qulacs import Observable
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN


class QNNRegressor(QNN):
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
        solver: Literal["Adam", "BFGS", "Nelder-Mead"] = "BFGS",
        cost: Literal["mse"] = "mse",
        do_x_scale: bool = True,
        do_y_scale: bool = True,
        x_norm_range=1.0,
        y_norm_range=0.7,
        callback=None,
    ) -> None:
        """
        :param circuit: Circuit to use in the learning.
        :param solver: Solver to use(Nelder-Mead is not recommended).
        :param cost: Cost function. MSE only for now. MSE computes squared sum after normalization.
        :param do_x_scale: Whether to scale x.
        :param do_x_scale: Whether to scale y.
        :param y_norm_range: Normalize y in [+-y_norm_range].
        :param callback: Callback function. Available only with Adam.
        """
        self.n_qubit = circuit.n_qubit
        self.circuit = circuit
        self.solver = solver
        self.cost = cost
        self.do_x_scale = do_x_scale
        self.do_y_scale = do_y_scale
        self.x_norm_range = x_norm_range
        self.y_norm_range = y_norm_range
        self.callback = callback
        self.observables = []

    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[float],
        maxiter: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
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
        if self.solver == "Nelder-Mead":
            result = minimize(
                self.cost_func,
                theta_init,
                args=(x_scaled, y_scaled),
                method=self.solver,
                options={"maxiter": maxiter},
            )
            loss = result.fun
            theta_opt = result.x
        elif self.solver == "BFGS":
            result = minimize(
                self.cost_func,
                theta_init,
                args=(x_scaled, y_scaled),
                method=self.solver,
                jac=self._cost_func_grad,
                options={"maxiter": maxiter},
            )
            loss = result.fun
            theta_opt = result.x
        elif self.solver == "Adam":
            pr_A = 0.02
            pr_Bi = 0.8
            pr_Bt = 0.995
            pr_ips = 0.0000001
            # Above is hyper parameters.
            Bix = 0
            Btx = 0

            moment = np.zeros(len(theta_init))
            vel = 0
            theta_now = theta_init
            maxiter *= len(x_scaled)
            for iter in range(0, maxiter, 5):
                grad = self._cost_func_grad(
                    theta_now,
                    x_scaled[iter % len(x_scaled) : iter % len(x_scaled) + 5],
                    y_scaled[iter % len(y_scaled) : iter % len(y_scaled) + 5],
                )
                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                if iter % len(x_scaled) < 5:
                    self.cost_func(theta_now, x_scaled, y_scaled)

            loss = self.cost_func(theta_now, x_scaled, y_scaled)
            theta_opt = theta_now
            if self.callback is not None:
                self.callback(theta_now)
        else:
            raise NotImplementedError

        return loss, theta_opt

    def predict(self, x_test: List[List[float]]) -> List[float]:
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

        if self.do_y_scale:
            y_pred = self.scale_y_scaler.inverse_transform(
                self._predict_inner(x_scaled)
            )
        else:
            y_pred = self._predict_inner(x_scaled)

        return y_pred

    def _predict_inner(self, x_scaled):
        res = []
        for x in x_scaled:
            state = self.circuit.run(x)
            r = [
                self.observables[i].get_expectation_value(state)
                for i in range(self.n_outputs)
            ]
            res.append(r)
        return np.array(res)

    def cost_func(self, theta, x_scaled, y_scaled):
        if self.cost == "mse":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)

            cost = mean_squared_error(y_pred, y_scaled)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _cost_func_grad(self, theta, x_scaled, y_scaled):
        self.circuit.update_parameters(theta)

        mto = self._predict_inner(x_scaled).copy()

        grad = np.zeros(len(theta))

        for h in range(len(x_scaled)):
            backobs = Observable(self.n_qubit)
            if self.n_outputs >= 2:
                for i in range(self.n_outputs):
                    backobs.add_operator(
                        (-y_scaled[h][i] + mto[h][i]) / self.n_outputs, "Z {i}"
                    )
            else:
                backobs.add_operator((-y_scaled[h] + mto[h][0]) / self.n_outputs, "Z 0")
            grad += self.circuit.backprop(x_scaled[h], backobs)

        self.circuit.update_parameters(theta)
        grad /= len(x_scaled)
        return grad
