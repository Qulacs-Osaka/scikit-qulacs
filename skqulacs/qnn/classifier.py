from __future__ import annotations

from typing import List, Optional

import numpy as np
from qulacs import Observable
from scipy.optimize import minimize
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN, _get_x_scale_param, _min_max_scaling


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
        solver: Literal["BFGS", "Nelder-Mead", "Adam"] = "BFGS",
        cost: Literal["log_loss"] = "log_loss",
        do_x_scale: bool = True,
        y_exp_ratio=2.2,
        callback=None,
    ) -> None:
        """
        :param circuit: Circuit to use in the learning.
        :param num_class: The number of classes; the number of qubits to measure.
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
        self.y_exp_ratio = y_exp_ratio
        self.callback = callback
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
        self.scale_x_param = _get_x_scale_param(x_train)
        self.scale_y_param = self.get_y_scale_param(y_train)

        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        else:
            x_scaled = x_train

        y_scaled = self.do_y_scale(y_train)

        # TODO: Extract solvers if the same one is used for classifier and regressor.
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
            maxiter *= len(x_train)
            for iter in range(0, maxiter, 5):
                grad = self._cost_func_grad(
                    theta_now,
                    x_scaled[iter % len(x_train) : iter % len(x_train) + 5],
                    y_scaled[iter % len(y_train) : iter % len(y_train) + 5],
                )
                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                if self.callback is not None:
                    self.callback(theta_now)

            loss = self.cost_func(theta_now, x_train, y_train)
            theta_opt = theta_now
        else:
            raise NotImplementedError
        return loss, theta_opt

    def predict(self, x_test: List[List[float]]):
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_test, self.scale_x_param)
        else:
            x_scaled = x_test

        y_pred = self.rev_y_scale(self._predict_inner(x_scaled))
        return y_pred

    def _predict_inner(self, x_list):
        res = []
        for x in x_list:
            state = self.circuit.run(x)
            r = [
                self.observables[i].get_expectation_value(state)
                for i in range(self.n_qubit)
            ]
            res.append(r)
        return np.array(res)

    # TODO: Extract cost function to outer class to accept other type of ones.
    def cost_func(self, theta, x_scaled, y_scaled):
        if self.cost == "log_loss":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)
            ypf = []
            for i in range(len(y_pred)):
                for j in range(len(self.scale_y_param[0])):
                    hid = self.scale_y_param[1][j]
                    wa = 0
                    for k in range(self.scale_y_param[0][j]):
                        wa += np.exp(self.y_exp_ratio * y_pred[i][hid + k])
                    for k in range(self.scale_y_param[0][j]):
                        ypf.append(np.exp(self.y_exp_ratio * y_pred[i][hid + k]) / wa)
            ysf = y_scaled.ravel()
            cost = 0
            for i in range(len(ysf)):
                if ysf[i] == 1:
                    cost -= np.log(ypf[i])
            cost /= len(ysf)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    # TODO: Extract following scaling operation as other class to change several scaling method.
    def get_y_scale_param(self, y):
        # Hold `y`'s maximum value.
        syurui = np.max(y, axis=0)
        syurui = syurui.astype(int)
        syurui = syurui + 1
        if not isinstance(syurui, np.ndarray):
            eee = syurui
            syurui = np.zeros(1, dtype=int)
            syurui[0] = eee
        rui = np.concatenate((np.zeros(1, dtype=int), syurui.cumsum()))
        return [syurui, rui]

    def do_y_scale(self, y):
        # Represent y as one-hot vector.
        # And handle multiple inputs.
        clsnum = int(self.scale_y_param[1][-1])
        res = np.zeros((len(y), clsnum), dtype=int)
        for i in range(len(y)):
            if y.ndim == 1:
                res[i][y[i]] = 1
            else:
                for j in range(len(y[i])):
                    res[i][y[i][j] + self.scale_y_param[1][j]] = 1
        return res

    def rev_y_scale(self, y_inr):
        res = np.zeros((len(y_inr), len(self.scale_y_param[0])), dtype=int)
        for i in range(len(y_inr)):
            for j in range(len(self.scale_y_param[0])):
                hid = self.scale_y_param[1][j]
                sai = -99998888
                arg = 0
                for k in range(self.scale_y_param[0][j]):
                    if sai < y_inr[i][hid + k]:
                        sai = y_inr[i][hid + k]
                        arg = k
                res[i][j] = arg
        return res

    def _cost_func_grad(self, theta, x_scaled, y_scaled):
        self.circuit.update_parameters(theta)

        mto = self._predict_inner(x_scaled).copy()
        grad = np.zeros(len(theta))
        for h in range(len(x_scaled)):
            for j in range(len(self.scale_y_param[0])):
                hid = self.scale_y_param[1][j]
                wa = 0
                for k in range(self.scale_y_param[0][j]):
                    wa += np.exp(self.y_exp_ratio * mto[h][hid + k])
                for k in range(self.scale_y_param[0][j]):
                    mto[h][hid + k] = np.exp(self.y_exp_ratio * mto[h][hid + k]) / wa

            backobs = Observable(self.n_qubit)
            for i in range(len(y_scaled[0])):
                backobs.add_operator(
                    (mto[h][i] - y_scaled[h][i]) * self.y_exp_ratio, f"Z {i}"
                )
            grad += self.circuit.backprop(x_scaled[h], backobs)

        self.circuit.update_parameters(theta)
        grad /= len(x_scaled)
        return grad
